"""Differentiable Go2 digital twin: rigid skeletal binding + PyTorch3D rendering.

Builds on go2_kinematics.Go2Kinematics. Each visual link owns a rigid mesh (bone);
forward kinematics gives the link's world transform; we transform that link's vertices
and compose all links into one scene via ``join_meshes_as_scene`` -- the same idiom the
drone renderer uses for static+dynamic scene compositing. The whole pipeline is
differentiable: gradients flow from rendered pixels (or vertex positions) back to the
joint angles ``q`` and the floating-base pose.

This is *rigid* skinning (one link = one rigid bone). True linear-blend skinning with
per-vertex weights is unnecessary for a rigid articulated robot.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    PerspectiveCameras, RasterizationSettings, MeshRasterizer, SoftPhongShader,
    PointLights, TexturesVertex, look_at_view_transform,
)
from pytorch3d.structures import Meshes, join_meshes_as_scene

import sys
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from go2_kinematics import Go2Kinematics  # noqa: E402
from go2_urdf import LEGS  # noqa: E402

OBJ_DIR = _HERE.parent / "assets" / "obj"

# Per-link-type RGB so the assembled robot is visually verifiable (binding check).
LINK_COLORS = {
    "base": (0.55, 0.57, 0.60),
    "hip":  (0.20, 0.55, 0.85),
    "thigh": (0.90, 0.62, 0.20),
    "calf": (0.30, 0.75, 0.40),
    "foot": (0.15, 0.15, 0.18),
}


def _link_type(link_name: str) -> str:
    if link_name == "base_link":
        return "base"
    for t in ("hip", "thigh", "calf", "foot"):
        if link_name.endswith(t):
            return t
    return "base"


class Go2Twin:
    """Differentiable Go2 visual digital twin."""

    def __init__(self, device="cpu", dtype=torch.float32):
        self.device = torch.device(device)
        self.dtype = dtype
        self.kin = Go2Kinematics.from_urdf(device=device, dtype=dtype)
        self.model = self.kin.model

        # Load each unique mesh once -> (verts (V,3), faces (F,3)).
        self._mesh_cache = {}
        # Visual links: name -> (mesh_key, visual_origin (4,4)).
        self.visual_links = {}
        for name, link in self.model.links.items():
            if link.visual is None or link.visual.mesh is None:
                continue
            key = link.visual.mesh
            if key not in self._mesh_cache:
                self._mesh_cache[key] = self._load_obj(key)
            from go2_kinematics import rpy_to_matrix, _homog
            R = rpy_to_matrix(torch.tensor(link.visual.origin_rpy, device=self.device, dtype=dtype))
            t = torch.tensor(link.visual.origin_xyz, device=self.device, dtype=dtype)
            self.visual_links[name] = (key, _homog(R, t))

    def _load_obj(self, key: str):
        m = load_objs_as_meshes([str(OBJ_DIR / f"{key}.obj")], device=self.device)
        return m.verts_packed().to(self.dtype), m.faces_packed()

    # ------------------------------------------------------------------ #
    # Skeletal binding -> single scene Meshes
    # ------------------------------------------------------------------ #
    def build_scene_mesh(self, q: torch.Tensor,
                         base_pos: Optional[torch.Tensor] = None,
                         base_R: Optional[torch.Tensor] = None) -> Meshes:
        if q.dim() == 1:
            q = q.unsqueeze(0)
        B = q.shape[0]
        T = self.kin.forward(q, base_pos, base_R)   # link -> (B,4,4)

        link_meshes = []
        for name, (key, T_vis) in self.visual_links.items():
            v_local, faces = self._mesh_cache[key]            # (V,3), (F,3)
            T_full = T[name] @ T_vis                          # (B,4,4)
            R = T_full[:, :3, :3]                             # (B,3,3)
            t = T_full[:, :3, 3]                              # (B,3)
            # v_world = v_local @ R^T + t   (batched, differentiable)
            v_world = torch.einsum("bij,vj->bvi", R, v_local) + t[:, None, :]
            rgb = torch.tensor(LINK_COLORS[_link_type(name)], device=self.device, dtype=self.dtype)
            tex = TexturesVertex(verts_features=rgb.expand(B, v_local.shape[0], 3))
            link_meshes.append(Meshes(verts=[v_world[b] for b in range(B)],
                                      faces=[faces for _ in range(B)], textures=tex))
        return join_meshes_as_scene(link_meshes)

    # ------------------------------------------------------------------ #
    # Rendering (URDF frame is used directly as the PyTorch3D world frame;
    # camera up = +Z keeps the robot upright). Kept self-contained & minimal.
    # ------------------------------------------------------------------ #
    def render(self, q, base_pos=None, base_R=None, *, image_size=512,
               dist=1.4, elev=18.0, azim=135.0, at=(0.0, 0.0, -0.18),
               return_tensor=False):
        mesh = self.build_scene_mesh(q, base_pos, base_R)
        B = len(mesh)
        at_t = torch.tensor([at], device=self.device, dtype=torch.float32).expand(B, 3)
        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim, at=at_t,
                                      up=((0.0, 0.0, 1.0),), device=self.device)
        cameras = PerspectiveCameras(focal_length=float(image_size), device=self.device,
                                     R=R, T=T,
                                     principal_point=((image_size / 2, image_size / 2),) * B,
                                     image_size=((image_size, image_size),) * B, in_ndc=False)
        raster = MeshRasterizer(cameras=cameras, raster_settings=RasterizationSettings(
            image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
            perspective_correct=True, bin_size=None, max_faces_per_bin=200000))
        lights = PointLights(device=self.device, location=[[1.5, 1.5, 2.0]])
        shader = SoftPhongShader(device=self.device, cameras=cameras, lights=lights)
        frags = raster(mesh)
        images = shader(frags, mesh)
        rgb = images[..., :3]
        depth = frags.zbuf[..., 0]
        if return_tensor:
            return rgb, depth
        return rgb.detach().cpu().numpy(), depth.detach().cpu().numpy()


if __name__ == "__main__":
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    twin = Go2Twin(device=dev, dtype=torch.float32)
    print(f"visual links: {len(twin.visual_links)}  unique meshes: {len(twin._mesh_cache)}")
    q = torch.zeros(1, 12, device=dev)
    mesh = twin.build_scene_mesh(q)
    print(f"scene mesh: V={mesh.verts_packed().shape[0]}  F={mesh.faces_packed().shape[0]}")
