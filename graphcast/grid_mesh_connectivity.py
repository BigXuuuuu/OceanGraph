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
"""Tools for converting from regular grids on a sphere, to triangular meshes."""

from graphcast import icosahedral_mesh
import numpy as np
import scipy
import trimesh
import haiku as hk
import jax.numpy as jnp
from typing import Tuple

def build_latlon_patches(
    grid_latitude: np.ndarray,
    grid_longitude: np.ndarray,
    patch_size_lat: int,
    patch_size_lon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
  """在规则 lat-lon 网格上构建 patch，并返回 patch 中心和映射关系。

  Args:
    grid_latitude: shape [num_lat] 的 1D 纬度数组。
    grid_longitude: shape [num_lon] 的 1D 经度数组。
    patch_size_lat: 纬向 patch 高度（多少格点合成一个 patch）。
    patch_size_lon: 经向 patch 宽度。

  Returns:
    patch_lat: [num_patches]，每个 patch 的中心纬度（简单取平均）
    patch_lon: [num_patches]，每个 patch 的中心经度
    grid_to_patch: [num_lat * num_lon]，展平后的每个格点 -> patch 索引
    num_patches: int，总的 patch 数
  """
  grid_latitude = np.asarray(grid_latitude, dtype=np.float32)
  grid_longitude = np.asarray(grid_longitude, dtype=np.float32)

  num_lat = grid_latitude.shape[0]
  num_lon = grid_longitude.shape[0]

  # 保存每个格点属于哪个 patch
  patch_indices_2d = -np.ones((num_lat, num_lon), dtype=np.int32)

  patch_centers_lat = []
  patch_centers_lon = []

  patch_id = 0
  for i in range(0, num_lat, patch_size_lat):
    lat_slice = slice(i, min(i + patch_size_lat, num_lat))
    sub_lat = grid_latitude[lat_slice]

    for j in range(0, num_lon, patch_size_lon):
      lon_slice = slice(j, min(j + patch_size_lon, num_lon))
      sub_lon = grid_longitude[lon_slice]

      # 这里简单取 patch 均值作为中心点，你之后也可以换成中间格点
      center_lat = float(sub_lat.mean())
      center_lon = float(sub_lon.mean())

      patch_centers_lat.append(center_lat)
      patch_centers_lon.append(center_lon)

      patch_indices_2d[lat_slice, lon_slice] = patch_id
      patch_id += 1

  grid_to_patch = patch_indices_2d.reshape(-1)  # [num_lat*num_lon]

  patch_lat = np.asarray(patch_centers_lat, dtype=np.float32)
  patch_lon = np.asarray(patch_centers_lon, dtype=np.float32)

  num_patches = int(patch_id)
  return patch_lat, patch_lon, grid_to_patch, num_patches



def _grid_lat_lon_to_coordinates(
    grid_latitude: np.ndarray, grid_longitude: np.ndarray) -> np.ndarray:
  """Lat [num_lat] lon [num_lon] to 3d coordinates [num_lat, num_lon, 3]."""
  # Convert to spherical coordinates phi and theta defined in the grid.
  # Each [num_latitude_points, num_longitude_points]
  phi_grid, theta_grid = np.meshgrid(
      np.deg2rad(grid_longitude),
      np.deg2rad(90 - grid_latitude))

  # [num_latitude_points, num_longitude_points, 3]
  # Note this assumes unit radius, since for now we model the earth as a
  # sphere of unit radius, and keep any vertical dimension as a regular grid.
  return np.stack(
      [np.cos(phi_grid)*np.sin(theta_grid),
       np.sin(phi_grid)*np.sin(theta_grid),
       np.cos(theta_grid)], axis=-1)


def radius_query_indices(
    *,
    grid_latitude: np.ndarray,
    grid_longitude: np.ndarray,
    mesh: icosahedral_mesh.TriangularMesh,
    radius: float) -> tuple[np.ndarray, np.ndarray]:
  """Returns mesh-grid edge indices for radius query.

  Args:
    grid_latitude: Latitude values for the grid [num_lat_points]
    grid_longitude: Longitude values for the grid [num_lon_points]
    mesh: Mesh object.
    radius: Radius of connectivity in R3. for a sphere of unit radius.

  Returns:
    tuple with `grid_indices` and `mesh_indices` indicating edges between the
    grid and the mesh such that the distances in a straight line (not geodesic)
    are smaller than or equal to `radius`.
    * grid_indices: Indices of shape [num_edges], that index into a
      [num_lat_points, num_lon_points] grid, after flattening the leading axes.
    * mesh_indices: Indices of shape [num_edges], that index into mesh.vertices.
  """

  # [num_grid_points=num_lat_points * num_lon_points, 3]
  grid_positions = _grid_lat_lon_to_coordinates(
      grid_latitude, grid_longitude).reshape([-1, 3])

  # [num_mesh_points, 3]
  mesh_positions = mesh.vertices
  kd_tree = scipy.spatial.cKDTree(mesh_positions)

  # [num_grid_points, num_mesh_points_per_grid_point]
  # Note `num_mesh_points_per_grid_point` is not constant, so this is a list
  # of arrays, rather than a 2d array.
  query_indices = kd_tree.query_ball_point(x=grid_positions, r=radius)

  grid_edge_indices = []
  mesh_edge_indices = []
  for grid_index, mesh_neighbors in enumerate(query_indices):
    grid_edge_indices.append(np.repeat(grid_index, len(mesh_neighbors)))
    mesh_edge_indices.append(mesh_neighbors)

  # [num_edges]
  grid_edge_indices = np.concatenate(grid_edge_indices, axis=0).astype(int)
  mesh_edge_indices = np.concatenate(mesh_edge_indices, axis=0).astype(int)

  return grid_edge_indices, mesh_edge_indices

def _latlon_to_cartesian(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
  """将纬度/经度（度）转换到单位球上的三维坐标，shape [..., 3]."""
  lat_rad = np.deg2rad(lat_deg)
  lon_rad = np.deg2rad(lon_deg)

  x = np.cos(lat_rad) * np.cos(lon_rad)
  y = np.cos(lat_rad) * np.sin(lon_rad)
  z = np.sin(lat_rad)
  return np.stack([x, y, z], axis=-1)


def radius_query_indices_for_patches(
    patch_latitude: np.ndarray,
    patch_longitude: np.ndarray,
    mesh: "icosahedral_mesh.Mesh",
    radius: float,
) -> Tuple[np.ndarray, np.ndarray]:
  """在单位球面上，对 patch 中心与 mesh 顶点做半径查询，得到 Patch-Grid2Mesh 边。

  Args:
    patch_latitude: [num_patches]，每个 patch 中心的纬度
    patch_longitude: [num_patches]，每个 patch 中心的经度
    mesh: icosahedral_mesh 的网格对象，要求有 vertices 属性 [num_mesh_nodes, 3]
    radius: float，单位球上的欧氏距离阈值

  Returns:
    patch_indices: [num_edges]，patch 节点索引（作为 senders）
    mesh_indices: [num_edges]，mesh 节点索引（作为 receivers）
  """
  patch_positions = _latlon_to_cartesian(patch_latitude, patch_longitude)
  mesh_vertices = np.asarray(mesh.vertices, dtype=np.float32)

  patch_indices_list = []
  mesh_indices_list = []

  r2 = float(radius ** 2)

  for patch_idx, pos in enumerate(patch_positions):
    # 与所有 mesh 顶点的平方距离
    d2 = np.sum((mesh_vertices - pos) ** 2, axis=-1)

    # 找到在 radius 内的 mesh 顶点
    neighbors = np.nonzero(d2 <= r2)[0]

    # 极端情况：半径太小，一个邻居都没有，就退化为最近的一个
    if neighbors.size == 0:
      neighbors = np.array([int(np.argmin(d2))], dtype=np.int32)

    mesh_indices_list.append(neighbors.astype(np.int32))
    patch_indices_list.append(
        np.full_like(neighbors, fill_value=patch_idx, dtype=np.int32)
    )

  patch_indices = np.concatenate(patch_indices_list, axis=0)
  mesh_indices = np.concatenate(mesh_indices_list, axis=0)

  print(f"Patch-Grid2Mesh edges: {patch_indices.shape[0]}")
  return patch_indices, mesh_indices



def in_mesh_triangle_indices(
    *,
    grid_latitude: np.ndarray,
    grid_longitude: np.ndarray,
    mesh: icosahedral_mesh.TriangularMesh) -> tuple[np.ndarray, np.ndarray]:
  """返回网格三角形中包含的网格点的网格网格边缘索引。

参数：
grid_latitude：网格的纬度值 [num_lat_points]
grid_longitude：网格的经度值 [num_lon_points]
mesh：网格对象。

返回：
具有 `grid_indices` 和 `mesh_indices` 的元组，表示网格与包含每个网格点的三角形的网格顶点之间的边缘。
边缘的数量始终为 num_lat_points * num_lon_points * 3
* grid_indices：形状为 [num_edges] 的索引，在展平主轴后，索引到
[num_lat_points, num_lon_points] 网格。
* mesh_indices：形状为 [num_edges] 的索引，索引到 mesh.vertices。
  """

  # 将网格的经纬度转换为三维坐标
  grid_positions = _grid_lat_lon_to_coordinates(
      grid_latitude, grid_longitude).reshape([-1, 3])

  # 创建三角形网格对象
  mesh_trimesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)

  # 查询每个网格点所在的三角形面
  _, _, query_face_indices = trimesh.proximity.closest_point(
      mesh_trimesh, grid_positions)

  triangle_vertex_indices = mesh.faces[query_face_indices]

  # 获取顶点坐标 [num_grid_points, 3, 3]
  triangle_vertices = mesh.vertices[triangle_vertex_indices]

  # 计算到每个顶点的距离 [num_grid_points, 3]
  distances = np.linalg.norm(
      triangle_vertices - grid_positions[:, np.newaxis, :],
      axis=2
  )

  # 获取距离排序索引并选择最近的两个 [num_grid_points, 2]
  closest_indices = np.argsort(distances, axis=1)[:, :2]

  # 提取最近的两个顶点索引 [num_grid_points, 2]
  closest_vertex_indices = np.take_along_axis(
      triangle_vertex_indices, closest_indices, axis=1)

  # 创建网格点索引（每个网格点重复两次）
  grid_indices = np.repeat(np.arange(grid_positions.shape[0]), 2)

  # 展平顶点索引 [num_grid_points * 2]
  mesh_indices = closest_vertex_indices.ravel()

  print(f"Mesh2Grid Edges: {grid_indices.shape[0]}")
  return grid_indices, mesh_indices


