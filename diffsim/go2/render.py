"""Chest-camera depth rendering from the same tensors used by Go2 physics."""

from __future__ import annotations

import math

import torch
from pytorch3d.renderer import MeshRasterizer, PerspectiveCameras, RasterizationSettings, look_at_view_transform
from pytorch3d.structures import Meshes
from pytorch3d.transforms import quaternion_to_matrix
from pytorch3d.utils import ico_sphere

from .model import Go2Model
from .scene import SHAPE_BOX, SHAPE_CAPSULE, SHAPE_SPHERE, Go2Scene
from .types import Go2EnvConfig, Go2State


_BOX_VERTICES = (
    (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
    (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
)
_BOX_FACES = (
    (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
    (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
    (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7),
)


def _append_mesh(verts, faces, new_verts, new_faces):
    offset = sum(item.shape[0] for item in verts)
    verts.append(new_verts)
    faces.append(new_faces + offset)


class Go2DepthRenderer:
    """Low-resolution PyTorch3D renderer; rendering is intentionally outside BPTT."""

    def __init__(self, model: Go2Model, config: Go2EnvConfig, device: torch.device):
        self.model = model
        self.config = config
        self.device = device
        self.dtype = model.spatial_inertia.dtype
        sphere = ico_sphere(1, device=device)
        self.sphere_vertices = sphere.verts_packed().to(self.dtype)
        self.sphere_faces = sphere.faces_packed()
        self.box_vertices = torch.tensor(_BOX_VERTICES, device=device, dtype=self.dtype)
        self.box_faces = torch.tensor(_BOX_FACES, device=device, dtype=torch.long)
        self._scene_key = None
        self._mesh = None

    def _terrain_mesh(self, scene: Go2Scene):
        nx, ny = 28, 20
        xs = torch.linspace(-2.0, 7.0, nx, device=self.device, dtype=self.dtype)
        ys = torch.linspace(-3.0, 3.0, ny, device=self.device, dtype=self.dtype)
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")
        xy = torch.stack((gx.flatten(), gy.flatten()), dim=-1)
        world_xy = xy[None].expand(scene.batch_size, -1, -1)
        height, _ = scene.terrain_height_and_normal(world_xy)
        vertices = torch.cat((world_xy, height[..., None]), dim=-1)
        faces = []
        for ix in range(nx - 1):
            for iy in range(ny - 1):
                a = ix * ny + iy
                b = a + ny
                faces.extend(((a, b + 1, b), (a, a + 1, b + 1)))
        return vertices, torch.tensor(faces, device=self.device, dtype=torch.long)

    def _build_scene_mesh(self, scene: Go2Scene) -> Meshes:
        terrain_vertices, terrain_faces = self._terrain_mesh(scene)
        all_vertices, all_faces = [], []
        for batch_index in range(scene.batch_size):
            verts = [terrain_vertices[batch_index]]
            faces = [terrain_faces]
            for obstacle in range(scene.obstacle_mask.shape[1]):
                if not bool(scene.obstacle_mask[batch_index, obstacle]):
                    continue
                kind = int(scene.obstacle_kind[batch_index, obstacle])
                position = scene.obstacle_position[batch_index, obstacle]
                rotation = scene.obstacle_rotation[batch_index, obstacle]
                size = scene.obstacle_size[batch_index, obstacle]
                if kind == SHAPE_BOX:
                    local = self.box_vertices * size
                    world = local @ rotation.T + position
                    _append_mesh(verts, faces, world, self.box_faces)
                elif kind == SHAPE_SPHERE:
                    world = self.sphere_vertices * size[0] + position
                    _append_mesh(verts, faces, world, self.sphere_faces)
                elif kind == SHAPE_CAPSULE:
                    for z in (-size[1], 0.0, size[1]):
                        center = position + rotation[:, 2] * z
                        world = self.sphere_vertices * size[0] + center
                        _append_mesh(verts, faces, world, self.sphere_faces)
            all_vertices.append(torch.cat(verts, dim=0))
            all_faces.append(torch.cat(faces, dim=0))
        return Meshes(verts=all_vertices, faces=all_faces)

    def camera_pose(self, state: Go2State):
        body_rotation = quaternion_to_matrix(state.base_quat)
        offset = state.base_pos.new_tensor(self.config.camera_pos_body)
        eye = state.base_pos + torch.einsum("bij,j->bi", body_rotation, offset)
        pitch_down = math.radians(-self.config.camera_pitch_deg)
        forward_body = state.base_pos.new_tensor((math.cos(pitch_down), 0.0, -math.sin(pitch_down)))
        forward_world = torch.einsum("bij,j->bi", body_rotation, forward_body)
        at = eye + forward_world
        up_world = torch.einsum(
            "bij,j->bi", body_rotation, state.base_pos.new_tensor((0.0, 0.0, 1.0))
        )
        return eye, at, up_world

    @torch.no_grad()
    def render(self, state: Go2State, scene: Go2Scene) -> torch.Tensor:
        mesh = self._build_scene_mesh(scene)
        eye, at, up = self.camera_pose(state)
        rotation, translation = look_at_view_transform(eye=eye, at=at, up=up, device=self.device)
        height, width = self.config.depth_height, self.config.depth_width
        focal = 0.5 * width / math.tan(math.radians(90.0) / 2.0)
        cameras = PerspectiveCameras(
            focal_length=((focal, focal),),
            principal_point=((width / 2.0, height / 2.0),),
            image_size=((height, width),),
            in_ndc=False,
            R=rotation,
            T=translation,
            device=self.device,
        )
        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=RasterizationSettings(
                image_size=(height, width),
                blur_radius=0.0,
                faces_per_pixel=1,
                perspective_correct=True,
                cull_backfaces=False,
            ),
        )
        depth = rasterizer(mesh).zbuf[..., 0]
        depth = torch.where(depth < 0.0, torch.full_like(depth, 6.0), depth.clamp(0.05, 6.0))
        return depth[:, None]
