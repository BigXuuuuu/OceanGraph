# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A predictor that runs multiple graph neural networks on mesh data.

Patched version with:
- Ocean LEVEL_OCEAN levels
- Patch-Grid2Mesh support (patch_size_lat, patch_size_lon in ModelConfig)
- Paper-aligned dynamic edge weights on Grid2Mesh, Mesh, and Mesh2Grid
"""

from typing import Any, Callable, Mapping, Optional
import jax.ops as jops

import chex
from graphcast import deep_typed_graph_net
from graphcast import grid_mesh_connectivity
from graphcast import icosahedral_mesh
from graphcast import losses
from graphcast import model_utils
from graphcast import predictor_base
from graphcast import typed_graph
from graphcast import xarray_jax
import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import xarray

Kwargs = Mapping[str, Any]
GNN = Callable[[jraph.GraphsTuple], jraph.GraphsTuple]

PRESSURE_LEVELS_ERA5_37 = (
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300,
    350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900,
    925, 950, 975, 1000)

PRESSURE_LEVELS_HRES_25 = (
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 400, 500, 600,
    700, 800, 850, 900, 925, 950, 1000)

PRESSURE_LEVELS_WEATHERBENCH_13 = (
    50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)

LEVEL_OCEAN = (
    0.494025, 2.645669, 5.078224
    # , 7.929560,
    # 11.405000, 15.810070, 21.598820,
    # 29.444730, 40.344050, 55.764290, 77.853850,
    # 92.326070, 130.666000, 186.125600,
    # 318.127400, 453.937700, 541.088900
)

PRESSURE_LEVELS = {
    3: LEVEL_OCEAN,
    13: PRESSURE_LEVELS_WEATHERBENCH_13,
    25: PRESSURE_LEVELS_HRES_25,
    37: PRESSURE_LEVELS_ERA5_37,
}

ALL_ATMOSPHERIC_VARS = (
    "thetao",
    "so",
    "uo",
    "vo",
)

TARGET_VARS = (
    "so",
    "siconc",
    "sithick",
    "thetao",
    "uo",
    "vo",
)

EXTERNAL_FORCING_VARS = (
    "usi",
    "vsi",
    "zos",
)

GENERATED_FORCING_VARS = (
    "year_progress_sin",
    "year_progress_cos",
    "day_progress_sin",
    "day_progress_cos",
)

FORCING_VARS = EXTERNAL_FORCING_VARS + GENERATED_FORCING_VARS

STATIC_VARS = (
    "land_sea_mask",
)


@chex.dataclass(frozen=True, eq=True)
class TaskConfig:
    """Defines inputs and targets on which a model is trained and/or evaluated."""
    input_variables: tuple[str, ...]
    target_variables: tuple[str, ...]
    forcing_variables: tuple[str, ...]
    pressure_levels: tuple[float, ...]
    input_duration: str


TASK = TaskConfig(
    input_variables=(TARGET_VARS + FORCING_VARS + STATIC_VARS),
    target_variables=TARGET_VARS,
    forcing_variables=FORCING_VARS,
    pressure_levels=LEVEL_OCEAN,
    input_duration="48h",
)


@chex.dataclass(frozen=True, eq=True)
class ModelConfig:
    """Defines the architecture of the GraphCast neural network."""
    resolution: float
    mesh_size: int
    latent_size: int
    gnn_msg_steps: int
    hidden_layers: int
    radius_query_fraction_edge_length: float
    mesh2grid_edge_normalization_factor: Optional[float] = None
    # Patch size (1 means no patching)
    patch_size_lat: int = 8
    patch_size_lon: int = 8


@chex.dataclass(frozen=True, eq=True)
class CheckPoint:
    params: dict[str, Any]
    model_config: ModelConfig
    task_config: TaskConfig
    description: str
    license: str


class GraphCast(predictor_base.Predictor):
    """GraphCast Predictor with Patch-Grid2Mesh and paper-aligned dynamic edge weights.

    节点类型：
      * mesh_nodes: icosahedral mesh 顶点
      * grid_nodes: 规则经纬网格点（或 patch 中心）

    图结构：
      * Grid2Mesh: grid/patch -> mesh 的双部图（radius query）
      * Mesh:      mesh -> mesh 的多层 GNN
      * Mesh2Grid: mesh -> grid 的双部图（三角剖分内插）
    """

    class DynamicEdgeWeights(hk.Module):
        """动态边权重学习模块（作用在 Grid2Mesh / Mesh2Grid 静态边特征上）."""

        def __init__(self, name: Optional[str] = None):
            super().__init__(name=name)
            # 简单 MLP: [in] -> 64 -> 1, sigmoid
            self._mlp = hk.Sequential([
                hk.Linear(64, name="dynamic_mlp_1"),
                jax.nn.swish,
                hk.Linear(1, name="dynamic_mlp_2"),
                jax.nn.sigmoid,
            ])

        def __call__(self, edge_features: jnp.ndarray) -> jnp.ndarray:
            # edge_features: [num_edges, feat_dim]
            return self._mlp(edge_features)  # [num_edges, 1]

    class PatchMLPPool(hk.Module):
        """先按 patch 做平均池化，再在 patch 特征上跑一个小 MLP."""

        def __init__(self,
                     hidden_size: int,
                     name: Optional[str] = None):
            super().__init__(name=name)
            self._hidden_size = hidden_size
            # 小 MLP：输入是 patch 特征，输出维度为 hidden_size
            self._mlp = hk.Sequential([
                hk.Linear(self._hidden_size),
                jax.nn.swish,
                hk.Linear(self._hidden_size),
            ])

        def __call__(self,
                     grid_node_features: chex.Array,
                     grid_to_patch: np.ndarray,
                     num_patches: int) -> chex.Array:
            """从格点特征得到 patch 特征并做 MLP。

            Args:
              grid_node_features: [num_grid_nodes, batch, in_channels]
              grid_to_patch:      [num_grid_nodes]，每个格点对应的 patch 索引（np 或 jnp）
              num_patches:        int，总 patch 数

            Returns:
              patch_features: [num_patches, batch, hidden_size]
            """
            # ------------- 1) 先做 patch 平均池化（不引入 MLP） -------------
            num_grid, batch, in_channels = grid_node_features.shape
            grid_to_patch = jnp.asarray(grid_to_patch, dtype=jnp.int32)

            # 把 (batch, channels) 合并，方便做 segment_sum:
            #   feats_flat: [num_grid_nodes, batch * in_channels]
            feats_flat = grid_node_features.reshape(num_grid, batch * in_channels)

            # 对每个 patch 求和
            summed = jops.segment_sum(
                feats_flat,
                grid_to_patch,
                num_segments=num_patches,        # 输出 [num_patches, batch * in_channels]
            )

            # 计数，用于做平均
            ones = jnp.ones((num_grid,), dtype=grid_node_features.dtype)
            counts = jops.segment_sum(ones, grid_to_patch, num_segments=num_patches)
            counts = jnp.where(counts > 0, counts, 1.0)

            means_flat = summed / counts[:, None]           # [num_patches, batch * in_channels]
            patch_feats = means_flat.reshape(num_patches, batch, in_channels)

            # ------------- 2) 再在 patch 级别跑小 MLP（显存压力大幅降低） -----
            # 把 patch 和 batch 合并成一个 batch 维度：
            #   patch_feats_2d: [num_patches * batch, in_channels]
            patch_feats_2d = patch_feats.reshape(num_patches * batch, in_channels)

            # MLP 映射到 hidden_size 维
            patch_feats_2d = self._mlp(patch_feats_2d)      # [num_patches * batch, hidden_size]

            # 再 reshape 回 [num_patches, batch, hidden_size]
            patch_feats_out = patch_feats_2d.reshape(num_patches, batch, self._hidden_size)
            return patch_feats_out

    class DynamicMeshProcessor(hk.Module):
        """与论文一致的 Mesh Processor。

        每一层都显式执行：
          1) 根据当前边状态 + 两端节点状态计算动态权重；
          2) 用动态权重缩放边更新；
          3) 聚合到接收节点并做节点更新；
          4) 对边和节点都做残差更新。
        """

        def __init__(self,
                     latent_size: int,
                     hidden_layers: int,
                     message_passing_steps: int,
                     use_layer_norm: bool = True,
                     name: Optional[str] = None):
            super().__init__(name=name)
            self._latent_size = latent_size
            self._hidden_layers = max(hidden_layers, 1)
            self._message_passing_steps = message_passing_steps
            self._use_layer_norm = use_layer_norm

        def _apply_mlp(self,
                       inputs: chex.Array,
                       output_size: int,
                       name: str) -> chex.Array:
            input_shape = inputs.shape[:-1]
            flat_inputs = inputs.reshape((-1, inputs.shape[-1]))
            widths = [self._latent_size] * self._hidden_layers + [output_size]
            flat_outputs = hk.nets.MLP(
                widths,
                activation=jax.nn.swish,
                activate_final=False,
                name=name,
            )(flat_inputs)
            return flat_outputs.reshape(input_shape + (output_size,))

        def __call__(self,
                     node_states: chex.Array,
                     edge_static_features: chex.Array,
                     senders: chex.Array,
                     receivers: chex.Array) -> chex.Array:
            num_nodes = node_states.shape[0]
            batch_size = node_states.shape[1]

            edge_states = self._apply_mlp(
                edge_static_features,
                self._latent_size,
                name="mesh_edge_embed",
            )
            edge_states = _add_batch_second_axis(
                edge_states.astype(node_states.dtype), batch_size)

            for step in range(self._message_passing_steps):
                sender_states = node_states[senders, :, :]
                receiver_states = node_states[receivers, :, :]

                dynamic_inputs = jnp.concatenate(
                    [edge_states, sender_states, receiver_states], axis=-1)
                dynamic_weights = self._apply_mlp(
                    dynamic_inputs,
                    1,
                    name=f"mesh_dynamic_weight_mlp_{step}",
                )
                dynamic_weights = jax.nn.sigmoid(dynamic_weights)

                edge_update_inputs = jnp.concatenate(
                    [edge_states, sender_states, receiver_states], axis=-1)
                raw_edge_updates = self._apply_mlp(
                    edge_update_inputs,
                    self._latent_size,
                    name=f"mesh_edge_update_mlp_{step}",
                )
                edge_updates = dynamic_weights * raw_edge_updates

                aggregated_messages = jops.segment_sum(
                    edge_updates,
                    receivers,
                    num_segments=num_nodes,
                )

                node_update_inputs = jnp.concatenate(
                    [node_states, aggregated_messages], axis=-1)
                node_updates = self._apply_mlp(
                    node_update_inputs,
                    self._latent_size,
                    name=f"mesh_node_update_mlp_{step}",
                )

                edge_states = edge_states + edge_updates
                node_states = node_states + node_updates

                if self._use_layer_norm:
                    edge_states = hk.LayerNorm(
                        axis=-1,
                        create_scale=True,
                        create_offset=True,
                        name=f"mesh_edge_layer_norm_{step}",
                    )(edge_states)
                    node_states = hk.LayerNorm(
                        axis=-1,
                        create_scale=True,
                        create_offset=True,
                        name=f"mesh_node_layer_norm_{step}",
                    )(node_states)

            return node_states


    def __init__(self, model_config: ModelConfig, task_config: TaskConfig):
        """Initializes the predictor."""
        self._spatial_features_kwargs = dict(
            add_node_positions=False,
            add_node_latitude=True,
            add_node_longitude=True,
            add_relative_positions=True,
            relative_longitude_local_coordinates=True,
            relative_latitude_local_coordinates=True,
        )
        self._model_config = model_config
        self._task_config = task_config

        # Multimesh
        self._meshes = (
            icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(
                splits=model_config.mesh_size))

        # Grid2Mesh encoder GNN
        self._grid2mesh_gnn = deep_typed_graph_net.DeepTypedGraphNet(
            embed_nodes=True,
            embed_edges=True,
            edge_latent_size=dict(grid2mesh=model_config.latent_size),
            node_latent_size=dict(
                mesh_nodes=model_config.latent_size,
                grid_nodes=model_config.latent_size),
            mlp_hidden_size=model_config.latent_size,
            mlp_num_hidden_layers=model_config.hidden_layers,
            num_message_passing_steps=1,
            use_layer_norm=True,
            include_sent_messages_in_node_update=False,
            activation="swish",
            f32_aggregation=True,
            aggregate_normalization=None,
            name="grid2mesh_gnn",
        )

        # Mesh processor GNN
        self._mesh_gnn = deep_typed_graph_net.DeepTypedGraphNet(
            embed_nodes=False,
            embed_edges=True,
            node_latent_size=dict(mesh_nodes=model_config.latent_size),
            edge_latent_size=dict(mesh=model_config.latent_size),
            mlp_hidden_size=model_config.latent_size,
            mlp_num_hidden_layers=model_config.hidden_layers,
            num_message_passing_steps=model_config.gnn_msg_steps,
            use_layer_norm=True,
            include_sent_messages_in_node_update=False,
            activation="swish",
            f32_aggregation=False,
            name="mesh_gnn",
        )

        num_surface_vars = len(
            set(task_config.target_variables) - set(ALL_ATMOSPHERIC_VARS))
        num_atmospheric_vars = len(
            set(task_config.target_variables) & set(ALL_ATMOSPHERIC_VARS))
        num_outputs = (num_surface_vars +
                       len(task_config.pressure_levels) * num_atmospheric_vars)

        # Mesh2Grid decoder GNN
        self._mesh2grid_gnn = deep_typed_graph_net.DeepTypedGraphNet(
            node_output_size=dict(grid_nodes=num_outputs),
            embed_nodes=False,
            embed_edges=True,
            edge_latent_size=dict(mesh2grid=model_config.latent_size),
            node_latent_size=dict(
                mesh_nodes=model_config.latent_size,
                grid_nodes=model_config.latent_size),
            mlp_hidden_size=model_config.latent_size,
            mlp_num_hidden_layers=model_config.hidden_layers,
            num_message_passing_steps=1,
            use_layer_norm=True,
            include_sent_messages_in_node_update=False,
            activation="swish",
            f32_aggregation=False,
            name="mesh2grid_gnn",
        )

        # Grid2Mesh radius on unit sphere
        self._query_radius = (_get_max_edge_distance(self._finest_mesh)
                              * model_config.radius_query_fraction_edge_length)
        self._mesh2grid_edge_normalization_factor = (
            model_config.mesh2grid_edge_normalization_factor
        )

        # Flags for lazy initialization
        self._initialized = False

        # Mesh properties (lazy)
        self._num_mesh_nodes: Optional[int] = None
        self._mesh_nodes_lat: Optional[np.ndarray] = None
        self._mesh_nodes_lon: Optional[np.ndarray] = None

        # Grid properties (lazy)
        self._grid_lat: Optional[np.ndarray] = None  # [num_lat]
        self._grid_lon: Optional[np.ndarray] = None  # [num_lon]
        self._num_grid_nodes: Optional[int] = None   # num_lat * num_lon
        self._grid_nodes_lat: Optional[np.ndarray] = None  # [num_grid_nodes]
        self._grid_nodes_lon: Optional[np.ndarray] = None  # [num_grid_nodes]

        # Patch-Grid properties (lazy)
        self._num_patch_nodes: Optional[int] = None          # num_patches
        self._patch_lat: Optional[np.ndarray] = None         # [num_patches]
        self._patch_lon: Optional[np.ndarray] = None         # [num_patches]
        self._patch_nodes_lat: Optional[np.ndarray] = None   # [num_patches]
        self._patch_nodes_lon: Optional[np.ndarray] = None   # [num_patches]
        self._grid_to_patch: Optional[np.ndarray] = None     # [num_grid_nodes]

        # Graph structures (lazy)
        self._grid2mesh_graph_structure: Optional[typed_graph.TypedGraph] = None
        self._mesh_graph_structure: Optional[typed_graph.TypedGraph] = None
        self._mesh2grid_graph_structure: Optional[typed_graph.TypedGraph] = None

    @property
    def _finest_mesh(self):
        return self._meshes[-1]

#     def __call__(self,
#                  inputs: xarray.Dataset,
#                  targets_template: xarray.Dataset,
#                  forcings: xarray.Dataset,
#                  is_training: bool = False,
#                  ) -> xarray.Dataset:
#         self._maybe_init(inputs)

#         # xarray (batch, time, lat, lon, level, vars) -> [num_grid_nodes, batch, channels]
#         grid_node_features = self._inputs_to_grid_node_features(inputs, forcings)

#         # Grid2Mesh encoder
#         latent_mesh_nodes, latent_grid_nodes = self._run_grid2mesh_gnn(
#             grid_node_features)

#         # Mesh processor
#         updated_latent_mesh_nodes = self._run_mesh_gnn(latent_mesh_nodes)

#         # Mesh2Grid decoder
#         output_grid_nodes = self._run_mesh2grid_gnn(
#             updated_latent_mesh_nodes, latent_grid_nodes)

#         # Flat grid outputs -> xarray prediction
#         return self._grid_node_outputs_to_prediction(
#             output_grid_nodes, targets_template)
    

    def __call__(self,
                 inputs: xarray.Dataset,
                 targets_template: xarray.Dataset,
                 forcings: xarray.Dataset,
                 is_training: bool = False,
                 ) -> xarray.Dataset:
        self._maybe_init(inputs)

        # 1) 先拿到格点上的输入特征: [num_grid_nodes, batch, channels]
        grid_node_features = self._inputs_to_grid_node_features(inputs, forcings)

        # 2) 如果启用 patch，则把格点特征聚合到 patch 上（这里已经用了 MLP 池化 B 方案）
        if self._using_patches():
            grid_node_features_for_gnn = self._pool_grid_to_patches(grid_node_features)
        else:
            grid_node_features_for_gnn = grid_node_features

        # 3) Grid2Mesh 只在 patch 节点（或原始 grid）上跑
        latent_mesh_nodes, latent_nodes = self._run_grid2mesh_gnn(
            grid_node_features_for_gnn)

        # 4) 如果启用 patch，需要把 patch latent 展开回格点 latent（注意：这里是 expand）
        if self._using_patches():
            latent_grid_nodes = self._expand_patches_to_grid(latent_nodes)
        else:
            latent_grid_nodes = latent_nodes

        # 5) Mesh processor
        updated_latent_mesh_nodes = self._run_mesh_gnn(latent_mesh_nodes)

        # 6) Mesh2Grid 使用完整格点的 latent
        output_grid_nodes = self._run_mesh2grid_gnn(
            updated_latent_mesh_nodes, latent_grid_nodes)

        # 7) 恢复成 xarray 预测场
        return self._grid_node_outputs_to_prediction(
            output_grid_nodes, targets_template)

    
    
    
    def loss_and_predictions(
        self,
        inputs: xarray.Dataset,
        targets: xarray.Dataset,
        forcings: xarray.Dataset,
    ) -> tuple[predictor_base.LossAndDiagnostics, xarray.Dataset]:
        """Forward + loss，用于训练和评估。"""
        predictions = self(
            inputs, targets_template=targets, forcings=forcings, is_training=True)
        loss = losses.weighted_mse_per_level(
            predictions, targets,
            per_variable_weights={
                "siconc": 0.1,
                "sithick": 0.1,
                "so": 1.0,
                "thetao": 1.0,
                "uo": 1.0,
                "vo": 1.0,
            })
        return loss, predictions

    def loss(
        self,
        inputs: xarray.Dataset,
        targets: xarray.Dataset,
        forcings: xarray.Dataset,
    ) -> predictor_base.LossAndDiagnostics:
        loss, _ = self.loss_and_predictions(inputs, targets, forcings)
        return loss

    def _maybe_init(self, sample_inputs: xarray.Dataset):
        """Inits everything that has a dependency on the input coordinates."""
        if not self._initialized:
            self._init_mesh_properties()
            self._init_grid_properties(
                grid_lat=np.asarray(sample_inputs.lat),
                grid_lon=np.asarray(sample_inputs.lon),
            )
            self._grid2mesh_graph_structure = self._init_grid2mesh_graph()
            self._mesh_graph_structure = self._init_mesh_graph()
            self._mesh2grid_graph_structure = self._init_mesh2grid_graph()
            self._initialized = True

    def _init_mesh_properties(self):
        """Inits static properties that have to do with mesh nodes."""
        self._num_mesh_nodes = self._finest_mesh.vertices.shape[0]
        mesh_phi, mesh_theta = model_utils.cartesian_to_spherical(
            self._finest_mesh.vertices[:, 0],
            self._finest_mesh.vertices[:, 1],
            self._finest_mesh.vertices[:, 2])
        mesh_nodes_lat, mesh_nodes_lon = model_utils.spherical_to_lat_lon(
            phi=mesh_phi, theta=mesh_theta)
        # Convert to f32 to ensure the lat/lon features aren't in f64.
        self._mesh_nodes_lat = mesh_nodes_lat.astype(np.float32)
        self._mesh_nodes_lon = mesh_nodes_lon.astype(np.float32)

    def _init_grid_properties(self, grid_lat: np.ndarray, grid_lon: np.ndarray):
        """Inits static properties that have to do with grid & patch nodes."""
        grid_lat = np.asarray(grid_lat, dtype=np.float32)
        grid_lon = np.asarray(grid_lon, dtype=np.float32)
        self._grid_lat = grid_lat
        self._grid_lon = grid_lon

        num_lat = grid_lat.shape[0]
        num_lon = grid_lon.shape[0]
        self._num_grid_nodes = int(num_lat * num_lon)

        # Initialize lat and lon for the flattened grid nodes.
        grid_nodes_lon, grid_nodes_lat = np.meshgrid(grid_lon, grid_lat)
        self._grid_nodes_lon = grid_nodes_lon.reshape([-1]).astype(np.float32)
        self._grid_nodes_lat = grid_nodes_lat.reshape([-1]).astype(np.float32)

        # -------- Patch-Grid 构建 --------
        patch_size_lat = getattr(self._model_config, "patch_size_lat", 1)
        patch_size_lon = getattr(self._model_config, "patch_size_lon", 1)

        if (patch_size_lat <= 1) and (patch_size_lon <= 1):
            # 不启用 patch，保持原始行为：每个格点自己是一个 patch
            self._num_patch_nodes = self._num_grid_nodes
            self._patch_lat = self._grid_nodes_lat
            self._patch_lon = self._grid_nodes_lon
            self._grid_to_patch = np.arange(self._num_grid_nodes, dtype=np.int32)
            self._patch_nodes_lat = self._patch_lat
            self._patch_nodes_lon = self._patch_lon
            print("[Patch-Grid] disabled (patch_size_lat/lon = 1)")
        else:
            (patch_lat, patch_lon,
             grid_to_patch, num_patches) = grid_mesh_connectivity.build_latlon_patches(
                 grid_latitude=self._grid_lat,
                 grid_longitude=self._grid_lon,
                 patch_size_lat=patch_size_lat,
                 patch_size_lon=patch_size_lon)

            self._num_patch_nodes = int(num_patches)
            self._patch_lat = patch_lat
            self._patch_lon = patch_lon
            self._grid_to_patch = grid_to_patch
            # Patch 中心的坐标就是 patch 节点坐标
            self._patch_nodes_lat = patch_lat
            self._patch_nodes_lon = patch_lon

            print(f"[Patch-Grid] num_grid_nodes = {self._num_grid_nodes}, "
                  f"num_patch_nodes = {self._num_patch_nodes}, "
                  f"patch_size_lat = {patch_size_lat}, "
                  f"patch_size_lon = {patch_size_lon}")
    
    def _using_patches(self) -> bool:
        return (self._num_patch_nodes is not None
            and self._num_patch_nodes != self._num_grid_nodes)
    
    def _expand_patches_to_grid(self, patch_latent: chex.Array) -> chex.Array:
        """[num_patch_nodes, batch, channels] -> [num_grid_nodes, batch, channels]."""
        if not self._using_patches():
            return patch_latent
    
        assert self._grid_to_patch is not None
        grid_to_patch = jnp.asarray(self._grid_to_patch, dtype=jnp.int32)
        # 直接用索引把每个格点对应的 patch latent 拿出来即可
        # patch_latent: [num_patch_nodes, batch, channels]
        # 结果:        [num_grid_nodes, batch, channels]
        return patch_latent[grid_to_patch, :, :]


    def _pool_grid_to_patches(self, grid_node_features: chex.Array) -> chex.Array:
        """[num_grid_nodes, batch, channels] -> [num_patch_nodes, batch, hidden_size].

        如果未启用 patch（patch_size_lat/lon = 1），则直接返回原始格点特征。
        """
        if not self._using_patches():
            # 不使用 patch，保持原来行为
            return grid_node_features

        assert self._grid_to_patch is not None
        num_patches = int(self._num_patch_nodes)

        # 创建 patch 级 MLP 池化模块（hidden_size 用 latent_size 比较自然）
        pooler = GraphCast.PatchMLPPool(
            hidden_size=self._model_config.latent_size,
            name="patch_pool_mlp",
        )

        # 返回 patch 级特征: [num_patches, batch, latent_size]
        return pooler(grid_node_features, self._grid_to_patch, num_patches)

    def _init_grid2mesh_graph(self) -> typed_graph.TypedGraph:
        """构建 Patch-Grid2Mesh 图（patch_size 为 1 时退化为原始 Grid2Mesh）。"""
        assert self._grid_lat is not None and self._grid_lon is not None
        assert self._mesh_nodes_lat is not None and self._mesh_nodes_lon is not None

        use_patch = (self._num_patch_nodes is not None and
                     self._num_patch_nodes != self._num_grid_nodes)

        if use_patch:
            # Patch-Grid2Mesh：用 patch 中心做 radius query
            assert self._patch_lat is not None and self._patch_lon is not None
            patch_indices, mesh_indices = (
                grid_mesh_connectivity.radius_query_indices_for_patches(
                    patch_latitude=self._patch_lat,
                    patch_longitude=self._patch_lon,
                    mesh=self._finest_mesh,
                    radius=self._query_radius)
            )
            senders = patch_indices       # patch 节点索引
            receivers = mesh_indices      # mesh 节点索引

            (senders_node_features, receivers_node_features,
             edge_features_np) = model_utils.get_bipartite_graph_spatial_features(
                 senders_node_lat=self._patch_nodes_lat,
                 senders_node_lon=self._patch_nodes_lon,
                 receivers_node_lat=self._mesh_nodes_lat,
                 receivers_node_lon=self._mesh_nodes_lon,
                 senders=senders,
                 receivers=receivers,
                 edge_normalization_factor=None,
                 **self._spatial_features_kwargs,
             )

            n_grid_node = np.array([self._num_patch_nodes], dtype=np.int32)
        else:
            # 退化为原始 Grid2Mesh：直接对所有格点做 radius query
            grid_indices, mesh_indices = grid_mesh_connectivity.radius_query_indices(
                grid_latitude=self._grid_lat,
                grid_longitude=self._grid_lon,
                mesh=self._finest_mesh,
                radius=self._query_radius)
            senders = grid_indices
            receivers = mesh_indices

            (senders_node_features, receivers_node_features,
             edge_features_np) = model_utils.get_bipartite_graph_spatial_features(
                 senders_node_lat=self._grid_nodes_lat,
                 senders_node_lon=self._grid_nodes_lon,
                 receivers_node_lat=self._mesh_nodes_lat,
                 receivers_node_lon=self._mesh_nodes_lon,
                 senders=senders,
                 receivers=receivers,
                 edge_normalization_factor=None,
                 **self._spatial_features_kwargs,
             )

            n_grid_node = np.array([self._num_grid_nodes], dtype=np.int32)

        # 这里只保留静态边特征；动态权重在 forward 时按论文公式实时计算。
        edge_features = jnp.asarray(edge_features_np, dtype=jnp.float32)

        # ---------- 统计信息 ----------
        num_edges = senders.shape[0]
        num_grid_nodes = int(n_grid_node[0])
        neighbor_counts = np.bincount(np.asarray(senders), minlength=num_grid_nodes)
        min_neighbors = int(neighbor_counts.min())
        max_neighbors = int(neighbor_counts.max())
        mean_neighbors = float(neighbor_counts.mean())
        print(
            f"[Grid2Mesh] num_grid_nodes = {num_grid_nodes}, "
            f"num_edges = {num_edges}, "
            f"neighbors per node: min={min_neighbors}, "
            f"mean={mean_neighbors:.2f}, max={max_neighbors}"
        )
        # -------------------------------

        n_mesh_node = np.array([self._num_mesh_nodes], dtype=np.int32)
        n_edge = np.array([receivers.shape[0]], dtype=np.int32)

        grid_node_set = typed_graph.NodeSet(
            n_node=n_grid_node, features=senders_node_features)
        mesh_node_set = typed_graph.NodeSet(
            n_node=n_mesh_node, features=receivers_node_features)
        edge_set = typed_graph.EdgeSet(
            n_edge=n_edge,
            indices=typed_graph.EdgesIndices(
                senders=senders, receivers=receivers),
            features=edge_features)

        nodes = {"grid_nodes": grid_node_set, "mesh_nodes": mesh_node_set}
        edges = {
            typed_graph.EdgeSetKey("grid2mesh", ("grid_nodes", "mesh_nodes")):
                edge_set
        }
        grid2mesh_graph = typed_graph.TypedGraph(
            context=typed_graph.Context(n_graph=np.array([1]), features=()),
            nodes=nodes,
            edges=edges)
        return grid2mesh_graph

    def _init_mesh_graph(self) -> typed_graph.TypedGraph:
        """Build Mesh graph."""
        merged_mesh = icosahedral_mesh.merge_meshes(self._meshes)

        # Work simply on the mesh edges.
        senders, receivers = icosahedral_mesh.faces_to_edges(merged_mesh.faces)

        # Structural features based on mesh nodes
        assert self._mesh_nodes_lat is not None and self._mesh_nodes_lon is not None
        node_features, edge_features = model_utils.get_graph_spatial_features(
            node_lat=self._mesh_nodes_lat,
            node_lon=self._mesh_nodes_lon,
            senders=senders,
            receivers=receivers,
            **self._spatial_features_kwargs,
        )

        n_mesh_node = np.array([self._num_mesh_nodes])
        n_edge = np.array([senders.shape[0]])
        mesh_node_set = typed_graph.NodeSet(
            n_node=n_mesh_node, features=node_features)
        edge_set = typed_graph.EdgeSet(
            n_edge=n_edge,
            indices=typed_graph.EdgesIndices(
                senders=senders, receivers=receivers),
            features=edge_features)
        nodes = {"mesh_nodes": mesh_node_set}
        edges = {
            typed_graph.EdgeSetKey("mesh", ("mesh_nodes", "mesh_nodes")): edge_set
        }
        mesh_graph = typed_graph.TypedGraph(
            context=typed_graph.Context(n_graph=np.array([1]), features=()),
            nodes=nodes,
            edges=edges)

        return mesh_graph

    def _init_mesh2grid_graph(self) -> typed_graph.TypedGraph:
        """Build Mesh2Grid graph."""
        grid_indices, mesh_indices = grid_mesh_connectivity.in_mesh_triangle_indices(
            grid_latitude=self._grid_lat,
            grid_longitude=self._grid_lon,
            mesh=self._finest_mesh)

        # Edges sending info from mesh to grid.
        senders = mesh_indices
        receivers = grid_indices

        # Structural bipartite features
        assert self._mesh_nodes_lat is not None and self._mesh_nodes_lon is not None
        (senders_node_features, receivers_node_features,
         edge_features) = model_utils.get_bipartite_graph_spatial_features(
             senders_node_lat=self._mesh_nodes_lat,
             senders_node_lon=self._mesh_nodes_lon,
             receivers_node_lat=self._grid_nodes_lat,
             receivers_node_lon=self._grid_nodes_lon,
             senders=senders,
             receivers=receivers,
             edge_normalization_factor=self._mesh2grid_edge_normalization_factor,
             **self._spatial_features_kwargs,
         )

        n_grid_node = np.array([self._num_grid_nodes])
        n_mesh_node = np.array([self._num_mesh_nodes])
        n_edge = np.array([senders.shape[0]])

        grid_node_set = typed_graph.NodeSet(
            n_node=n_grid_node, features=receivers_node_features)
        mesh_node_set = typed_graph.NodeSet(
            n_node=n_mesh_node, features=senders_node_features)
        edge_set = typed_graph.EdgeSet(
            n_edge=n_edge,
            indices=typed_graph.EdgesIndices(
                senders=senders, receivers=receivers),
            features=edge_features)
        nodes = {"grid_nodes": grid_node_set, "mesh_nodes": mesh_node_set}
        edges = {
            typed_graph.EdgeSetKey("mesh2grid", ("mesh_nodes", "grid_nodes")):
                edge_set
        }
        mesh2grid_graph = typed_graph.TypedGraph(
            context=typed_graph.Context(n_graph=np.array([1]), features=()),
            nodes=nodes,
            edges=edges)
        return mesh2grid_graph

    def _run_grid2mesh_gnn(self, grid_node_features: chex.Array,
                           ) -> tuple[chex.Array, chex.Array]:
        """Runs the grid2mesh_gnn, extracting latent mesh and grid nodes."""
        batch_size = grid_node_features.shape[1]

        grid2mesh_graph = self._grid2mesh_graph_structure
        assert grid2mesh_graph is not None
        grid_nodes = grid2mesh_graph.nodes["grid_nodes"]
        mesh_nodes = grid2mesh_graph.nodes["mesh_nodes"]

        new_grid_nodes = grid_nodes._replace(
            features=jnp.concatenate([
                grid_node_features,
                _add_batch_second_axis(
                    grid_nodes.features.astype(grid_node_features.dtype),
                    batch_size)
            ], axis=-1))

        # Mesh 节点补零，使得输入维度和 grid 节点对齐
        dummy_mesh_node_features = jnp.zeros(
            (self._num_mesh_nodes,) + grid_node_features.shape[1:],
            dtype=grid_node_features.dtype)
        new_mesh_nodes = mesh_nodes._replace(
            features=jnp.concatenate([
                dummy_mesh_node_features,
                _add_batch_second_axis(
                    mesh_nodes.features.astype(dummy_mesh_node_features.dtype),
                    batch_size)
            ], axis=-1))

        grid2mesh_edges_key = grid2mesh_graph.edge_key_by_name("grid2mesh")
        edges = grid2mesh_graph.edges[grid2mesh_edges_key]

        static_edge_features = edges.features.astype(dummy_mesh_node_features.dtype)
        dynamic_weights = GraphCast.DynamicEdgeWeights(
            name="grid2mesh_dynamic")(static_edge_features)
        edge_features = jnp.concatenate(
            [static_edge_features, dynamic_weights], axis=-1)

        new_edges = edges._replace(
            features=_add_batch_second_axis(edge_features, batch_size))

        input_graph = grid2mesh_graph._replace(
            edges={grid2mesh_edges_key: new_edges},
            nodes={
                "grid_nodes": new_grid_nodes,
                "mesh_nodes": new_mesh_nodes
            })

        # Run the GNN.
        grid2mesh_out = self._grid2mesh_gnn(input_graph)
        latent_mesh_nodes = grid2mesh_out.nodes["mesh_nodes"].features
        latent_grid_nodes = grid2mesh_out.nodes["grid_nodes"].features
        return latent_mesh_nodes, latent_grid_nodes

    def _run_mesh_gnn(self, latent_mesh_nodes: chex.Array) -> chex.Array:
        """论文一致的 Mesh Processor：每层动态计算边权并缩放边更新。"""
        mesh_graph = self._mesh_graph_structure
        assert mesh_graph is not None
        mesh_edges_key = mesh_graph.edge_key_by_name("mesh")
        edges = mesh_graph.edges[mesh_edges_key]

        processor = GraphCast.DynamicMeshProcessor(
            latent_size=self._model_config.latent_size,
            hidden_layers=self._model_config.hidden_layers,
            message_passing_steps=self._model_config.gnn_msg_steps,
            use_layer_norm=True,
            name="mesh_dynamic_processor",
        )
        return processor(
            node_states=latent_mesh_nodes,
            edge_static_features=edges.features.astype(latent_mesh_nodes.dtype),
            senders=edges.indices.senders,
            receivers=edges.indices.receivers,
        )

    def _run_mesh2grid_gnn(
        self,
        updated_latent_mesh_nodes: chex.Array,
        latent_grid_nodes: chex.Array,
    ) -> chex.Array:
        """Runs the mesh2grid_gnn, extracting the output grid nodes."""
        batch_size = updated_latent_mesh_nodes.shape[1]

        mesh2grid_graph = self._mesh2grid_graph_structure
        assert mesh2grid_graph is not None
        mesh_nodes = mesh2grid_graph.nodes["mesh_nodes"]
        grid_nodes = mesh2grid_graph.nodes["grid_nodes"]
        new_mesh_nodes = mesh_nodes._replace(features=updated_latent_mesh_nodes)
        new_grid_nodes = grid_nodes._replace(features=latent_grid_nodes)
        mesh2grid_key = mesh2grid_graph.edge_key_by_name("mesh2grid")
        edges = mesh2grid_graph.edges[mesh2grid_key]

        static_edge_features = edges.features.astype(latent_grid_nodes.dtype)
        dynamic_weights = GraphCast.DynamicEdgeWeights(
            name="mesh2grid_dynamic")(static_edge_features)
        edge_features = jnp.concatenate(
            [static_edge_features, dynamic_weights], axis=-1)

        new_edges = edges._replace(
            features=_add_batch_second_axis(edge_features, batch_size))

        input_graph = mesh2grid_graph._replace(
            edges={mesh2grid_key: new_edges},
            nodes={
                "mesh_nodes": new_mesh_nodes,
                "grid_nodes": new_grid_nodes
            })

        # Run the GNN.
        output_graph = self._mesh2grid_gnn(input_graph)
        output_grid_nodes = output_graph.nodes["grid_nodes"].features

        return output_grid_nodes

    def _inputs_to_grid_node_features(
        self,
        inputs: xarray.Dataset,
        forcings: xarray.Dataset,
    ) -> chex.Array:
        """xarrays -> [num_grid_nodes, batch, num_channels]."""

        # xarray `Dataset` (batch, time, lat, lon, level, multiple vars)
        # to xarray `DataArray` (batch, lat, lon, channels)
        stacked_inputs = model_utils.dataset_to_stacked(inputs)
        stacked_forcings = model_utils.dataset_to_stacked(forcings)
        stacked_inputs = xarray.concat(
            [stacked_inputs, stacked_forcings], dim="channels")

        # xarray `DataArray` (batch, lat, lon, channels)
        # to single numpy array with shape [lat_lon_node, batch, channels]
        grid_xarray_lat_lon_leading = model_utils.lat_lon_to_leading_axes(
            stacked_inputs)
        data = xarray_jax.unwrap(grid_xarray_lat_lon_leading.data)
        return data.reshape((-1,) + data.shape[2:])

    def _grid_node_outputs_to_prediction(
        self,
        grid_node_outputs: chex.Array,
        targets_template: xarray.Dataset,
    ) -> xarray.Dataset:
        """[num_grid_nodes, batch, num_outputs] -> xarray."""
        assert self._grid_lat is not None and self._grid_lon is not None
        grid_shape = (self._grid_lat.shape[0], self._grid_lon.shape[0])
        grid_outputs_lat_lon_leading = grid_node_outputs.reshape(
            grid_shape + grid_node_outputs.shape[1:])
        dims = ("lat", "lon", "batch", "channels")
        grid_xarray_lat_lon_leading = xarray_jax.DataArray(
            data=grid_outputs_lat_lon_leading,
            dims=dims)
        grid_xarray = model_utils.restore_leading_axes(
            grid_xarray_lat_lon_leading)

        # xarray `DataArray` (batch, lat, lon, channels)
        # to xarray `Dataset` (batch, one time step, lat, lon, level, multiple vars)
        return model_utils.stacked_to_dataset(
            grid_xarray.variable, targets_template)


def _add_batch_second_axis(data: chex.Array, batch_size: int) -> chex.Array:
    """data [N, C] -> [N, batch, C] by broadcasting batch axis."""
    assert data.ndim == 2
    ones = jnp.ones([batch_size, 1], dtype=data.dtype)
    return data[:, None] * ones  # [N, batch, C]


def _get_max_edge_distance(mesh) -> float:
    senders, receivers = icosahedral_mesh.faces_to_edges(mesh.faces)
    edge_distances = np.linalg.norm(
        mesh.vertices[senders] - mesh.vertices[receivers], axis=-1)
    return float(edge_distances.max())
