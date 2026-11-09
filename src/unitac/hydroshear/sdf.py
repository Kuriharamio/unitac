"""Exact triangle-mesh signed distance using HydroShear's fused CUDA kernel.

Derived from HydroShear by MMintLab (MIT); see ``licenses/HydroShear-MIT.txt``.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

_POINT_SDF_KERNEL = r"""
// Per-point exact signed distance to a triangle mesh.
//
// Each thread owns one query point: it loops over all cached triangles for
// the closest point (Ericson, "Real-Time Collision Detection") and then a
// second loop for the inside/outside parity (Moller-Trumbore ray along +x).
// The base triangles are uploaded once; every per-step query is one launch.
extern "C" __global__ void point_mesh_sdf_kernel(
    const float* __restrict__ tris,    // (T, 9) base triangles
    const float* __restrict__ pts,     // (P, 3) query points
    float* __restrict__ sdf,           // (P,) signed distance
    float* __restrict__ grad,          // (P, 3) unit gradient toward higher SDF
    const float* __restrict__ tnorm,   // (T, 3) unit face normals
    int T, int P, int compute_grad)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    const float px = pts[(size_t)p * 3];
    const float py = pts[(size_t)p * 3 + 1];
    const float pz = pts[(size_t)p * 3 + 2];

    float best = 1e30f;
    float qx = 0.f, qy = 0.f, qz = 0.f;
    int bt = -1;
    for (int t = 0; t < T; ++t) {
        const float* tr = tris + (size_t)t * 9;
        float ax = tr[0], ay = tr[1], az = tr[2];
        float bx = tr[3], by = tr[4], bz = tr[5];
        float cx = tr[6], cy = tr[7], cz = tr[8];
        float abx = bx - ax, aby = by - ay, abz = bz - az;
        float acx = cx - ax, acy = cy - ay, acz = cz - az;
        float apx = px - ax, apy = py - ay, apz = pz - az;
        float d1 = abx * apx + aby * apy + abz * apz;
        float d2 = acx * apx + acy * apy + acz * apz;
        float nx = ax, ny = ay, nz = az;  // candidate closest point
        if (d1 <= 0.f && d2 <= 0.f) {
            // region A: vertex A
        } else {
            float bpx = px - bx, bpy = py - by, bpz = pz - bz;
            float d3 = abx * bpx + aby * bpy + abz * bpz;
            float d4 = acx * bpx + acy * bpy + acz * bpz;
            if (d3 >= 0.f && d4 <= d3) {
                nx = bx; ny = by; nz = bz;          // region B
            } else {
                float vc = d1 * d4 - d3 * d2;
                if (vc <= 0.f && d1 >= 0.f && d3 <= 0.f) {
                    float v = d1 / (d1 - d3);       // edge AB
                    nx = ax + v * abx; ny = ay + v * aby; nz = az + v * abz;
                } else {
                    float cpx = px - cx, cpy = py - cy, cpz = pz - cz;
                    float d5 = abx * cpx + aby * cpy + abz * cpz;
                    float d6 = acx * cpx + acy * cpy + acz * cpz;
                    if (d6 >= 0.f && d5 <= d6) {
                        nx = cx; ny = cy; nz = cz;  // region C
                    } else {
                        float vb = d5 * d2 - d1 * d6;
                        if (vb <= 0.f && d2 >= 0.f && d6 <= 0.f) {
                            float w = d2 / (d2 - d6);  // edge AC
                            nx = ax + w * acx; ny = ay + w * acy; nz = az + w * acz;
                        } else {
                            float va = d3 * d6 - d5 * d4;
                            if (va <= 0.f && (d4 - d3) >= 0.f && (d5 - d6) >= 0.f) {
                                float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));  // edge BC
                                nx = bx + w * (cx - bx);
                                ny = by + w * (cy - by);
                                nz = bz + w * (cz - bz);
                            } else {
                                float denom = 1.f / (va + vb + vc);  // inside face
                                float v = vb * denom;
                                float w = vc * denom;
                                nx = ax + abx * v + acx * w;
                                ny = ay + aby * v + acy * w;
                                nz = az + abz * v + acz * w;
                            }
                        }
                    }
                }
            }
        }
        float dx = px - nx, dy = py - ny, dz = pz - nz;
        float dd = dx * dx + dy * dy + dz * dz;
        if (dd < best) { best = dd; qx = nx; qy = ny; qz = nz; bt = t; }
    }
    float dist = sqrtf(best);

    // Inside/outside by ray parity along +x.
    int hits = 0;
    for (int t = 0; t < T; ++t) {
        const float* tr = tris + (size_t)t * 9;
        float ax = tr[0], ay = tr[1], az = tr[2];
        float bx = tr[3], by = tr[4], bz = tr[5];
        float cx = tr[6], cy = tr[7], cz = tr[8];
        float e1x = bx - ax, e1y = by - ay, e1z = bz - az;
        float e2x = cx - ax, e2y = cy - ay, e2z = cz - az;
        // dir = (1, 0, 0): h = cross(dir, e2) = (0, -e2z, e2y)
        float hx = 0.f, hy = -e2z, hz = e2y;
        float det = e1x * hx + e1y * hy + e1z * hz;
        if (fabsf(det) < 1e-12f) continue;
        float inv = 1.f / det;
        float sx = px - ax, sy = py - ay, sz = pz - az;
        float u = (sx * hx + sy * hy + sz * hz) * inv;
        if (u < -1e-6f || u > 1.f + 1e-6f) continue;
        float qvx = sy * e1z - sz * e1y;
        float qvy = sz * e1x - sx * e1z;
        float qvz = sx * e1y - sy * e1x;
        float v = qvx * inv;  // dot(dir=(1,0,0), cross(s,e1))
        if (v < -1e-6f || u + v > 1.f + 1e-6f) continue;
        float tt = (e2x * qvx + e2y * qvy + e2z * qvz) * inv;
        if (tt > 1e-7f) hits++;
    }
    float sgn = (hits & 1) ? -1.f : 1.f;
    sdf[p] = sgn * dist;

    if (compute_grad) {
        // Unit gradient toward higher SDF (outward normals on / near the surface).
        if (dist > 1e-4f) {
            grad[(size_t)p * 3] = sgn * (px - qx) / dist;
            grad[(size_t)p * 3 + 1] = sgn * (py - qy) / dist;
            grad[(size_t)p * 3 + 2] = sgn * (pz - qz) / dist;
        } else if (bt >= 0) {
            grad[(size_t)p * 3] = tnorm[(size_t)bt * 3];
            grad[(size_t)p * 3 + 1] = tnorm[(size_t)bt * 3 + 1];
            grad[(size_t)p * 3 + 2] = tnorm[(size_t)bt * 3 + 2];
        } else {
            grad[(size_t)p * 3] = grad[(size_t)p * 3 + 1] = grad[(size_t)p * 3 + 2] = 0.f;
        }
    }
}
"""


@lru_cache(maxsize=16)
def _point_sdf_kernel():
    import cupy as cp

    return cp.RawKernel(_POINT_SDF_KERNEL, "point_mesh_sdf_kernel")


def _to_cupy(tensor: torch.Tensor):
    import cupy as cp

    if tensor.is_cuda:
        return cp.asarray(cp.from_dlpack(tensor.contiguous()), dtype=cp.float32)
    return cp.asarray(tensor.detach().cpu().numpy(), dtype=cp.float32)


def _from_cupy(array, device: torch.device) -> torch.Tensor:
    if device.type == "cuda":
        return torch.from_dlpack(array).to(device=device)
    return torch.from_numpy(array.get())


class MeshSDFSensor:
    """Query signed distances to a closed triangle mesh at torch points."""

    def __init__(self, triangles: np.ndarray) -> None:
        triangles = np.asarray(triangles, dtype=np.float32)
        if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or not triangles.size:
            raise ValueError("triangles must have shape (N, 3, 3), N > 0")
        if not np.isfinite(triangles).all():
            raise ValueError("triangles must be finite")
        self._triangles = np.ascontiguousarray(triangles.reshape(-1, 9))
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        lengths[lengths == 0.0] = 1.0
        self._normals = np.ascontiguousarray(normals / lengths, dtype=np.float32)
        self._triangles_gpu = None
        self._normals_gpu = None
        self._distances_gpu = None
        self._unused_gradient_gpu = None

    def _gpu_geometry(self):
        import cupy as cp

        if self._triangles_gpu is None:
            self._triangles_gpu = cp.asarray(self._triangles)
            self._normals_gpu = cp.asarray(self._normals)
        return self._triangles_gpu, self._normals_gpu

    def prepare(self, point_count: int) -> None:
        """Allocate fixed-size query buffers on the cold path."""
        import cupy as cp

        if point_count <= 0:
            raise ValueError("SDF point_count must be positive")
        self._gpu_geometry()
        self._distances_gpu = cp.empty(point_count, dtype=cp.float32)
        self._unused_gradient_gpu = cp.empty(1, dtype=cp.float32)

    def get_distance(self, points: torch.Tensor) -> torch.Tensor:
        """Return signed distances shaped like ``points.shape[:-1]``."""
        if points.shape[-1] != 3:
            raise ValueError("points must end in dimension 3")

        shape = points.shape
        flat_points = _to_cupy(points.reshape(-1, 3))
        triangles, normals = self._gpu_geometry()
        if self._distances_gpu is None or self._distances_gpu.size != flat_points.shape[0]:
            self.prepare(flat_points.shape[0])
        distances = self._distances_gpu
        unused_gradient = self._unused_gradient_gpu
        kernel = _point_sdf_kernel()
        block = 256
        kernel(
            ((flat_points.shape[0] + block - 1) // block,),
            (block,),
            (
                triangles,
                flat_points,
                distances,
                unused_gradient,
                normals,
                triangles.shape[0],
                flat_points.shape[0],
                0,
            ),
        )
        return _from_cupy(distances, points.device).reshape(shape[:-1])
