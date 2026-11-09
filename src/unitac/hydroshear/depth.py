"""HydroShear scanline depth rasterizer.

Derived from HydroShear by MMintLab (MIT); see ``licenses/HydroShear-MIT.txt``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

DEPTH_H, DEPTH_W = 175, 100
CAMERA_W, CAMERA_H = 240, 320
VFOV_DEG = 72.146676
RENDER_W, RENDER_H = 104, 175
CAMERA_DIST_M = 0.0267
CAMERA_NEAR_M = 0.0001
CAMERA_TOP_Z_M = 0.0
CAM_Z = -CAMERA_DIST_M
CAMERA_TAN_V = np.tan(np.deg2rad(VFOV_DEG) / 2.0)
CAMERA_TAN_H = CAMERA_TAN_V * CAMERA_W / CAMERA_H
CAMERA_PRINCIPAL_U = 0.0
CAMERA_PRINCIPAL_V = 0.0

_RASTER_KERNELS = r"""
// Fused per-(env, triangle) scanline rasterizer.
extern "C" __global__ void raster_depth_kernel(
    const float* __restrict__ base,   // (N, 9) base world triangles
    const float* __restrict__ rot,    // (B, 9) row-major 3x3 rotation
    const float* __restrict__ trans,  // (B, 3) translation (m)
    unsigned int* __restrict__ zbuf,  // (B, H*W) uint32, sentinel 0xFFFFFFFF
    int N, int W, int H, float cam_z, float tan_h, float tan_v,
    float principal_u, float principal_v, float near_plane)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    if (i >= N) return;
    const float* T = base + (size_t)i * 9;
    const float* r = rot + (size_t)b * 9;
    const float* tt = trans + (size_t)b * 3;
    float wx[3], wy[3], wz[3], u[3], v[3], zc[3];
    #pragma unroll
    for (int k = 0; k < 3; ++k) {
        wx[k] = r[0]*T[k*3] + r[1]*T[k*3+1] + r[2]*T[k*3+2] + tt[0];
        wy[k] = r[3]*T[k*3] + r[4]*T[k*3+1] + r[5]*T[k*3+2] + tt[1];
        wz[k] = r[6]*T[k*3] + r[7]*T[k*3+1] + r[8]*T[k*3+2] + tt[2];
        float z = wz[k] - cam_z;
        if (z <= near_plane) return;
        zc[k] = z;
        u[k] = (wx[k] / z / tan_h + 1.f) * 0.5f * W - 0.5f + principal_u;
        v[k] = (wy[k] / z / tan_v + 1.f) * 0.5f * H - 0.5f + principal_v;
    }

    float e1x = wx[1]-wx[0], e1y = wy[1]-wy[0], e1z = zc[1]-zc[0];
    float e2x = wx[2]-wx[0], e2y = wy[2]-wy[0], e2z = zc[2]-zc[0];
    float nx = e1y*e2z - e1z*e2y;
    float ny = e1z*e2x - e1x*e2z;
    float nz = e1x*e2y - e1y*e2x;
    float num = nx*wx[0] + ny*wy[0] + nz*zc[0];
    float len2 = nx*nx + ny*ny + nz*nz;
    if (len2 < 1e-18f) return;

    float vmin = fminf(v[0], fminf(v[1], v[2]));
    float vmax = fmaxf(v[0], fmaxf(v[1], v[2]));
    int v0i = (int)fmaxf(floorf(vmin) - 1.f, 0.f);
    int v1i = (int)fminf(ceilf(vmax) + 1.f, (float)(H - 1));
    size_t pbase = (size_t)b * W * H;
    for (int vv = v0i; vv <= v1i; ++vv) {
        float vc = (float)vv;
        float ul = 1e30f, ur = -1e30f;
        int hits = 0;
        #pragma unroll
        for (int e = 0; e < 3; ++e) {
            int k0 = e, k1 = (e + 1) % 3;
            float va = v[k0], vb = v[k1];
            float lo = fminf(va, vb), hi = fmaxf(va, vb);
            if (vc < lo || vc > hi) continue;
            float dv = vb - va;
            if (fabsf(dv) < 1e-9f) continue;
            float uc = u[k0] + (u[k1] - u[k0]) * (vc - va) / dv;
            ul = fminf(ul, uc); ur = fmaxf(ur, uc);
            hits++;
        }
        if (hits < 2) continue;
        int u0 = (int)fmaxf(ceilf(ul - 1e-4f), 0.f);
        int u1 = (int)fminf(floorf(ur + 1e-4f), (float)(W - 1));
        float av = tan_v * (2.f * (vc + 0.5f) / H - 1.f);
        size_t row = pbase + (size_t)vv * W;
        for (int uu = u0; uu <= u1; ++uu) {
            float au = tan_h * (2.f * (uu + 0.5f - principal_u) / W - 1.f);
            float av_shift = tan_v * (2.f * (vc + 0.5f - principal_v) / H - 1.f);
            float denom = nx*au + ny*av_shift + nz;
            if (fabsf(denom) < 1e-12f) continue;
            float z = cam_z + num / denom;
            unsigned int key =
                (z < 0.f) ? ~__float_as_uint(z)
                          : (__float_as_uint(z) | 0x80000000u);
            atomicMin(&zbuf[row + uu], key);
        }
    }
}
"""


_FINISH_KERNELS = r"""
extern "C" __global__ void finish_depth_kernel(
    const unsigned int* __restrict__ zbuf,  // (B, H, W) encoded z keys
    const int* __restrict__ src0,       // (Wout,) first source column
    const int* __restrict__ src1,       // (Wout,) second source column (-1 none)
    const float* __restrict__ w0,       // (Wout,)
    const float* __restrict__ w1,       // (Wout,)
    float* __restrict__ out,            // (B, H, Wout) mm indentation <= 0
    int W, int H, int Wout, int B, float gel_top_z)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * Wout;
    if (idx >= total) return;
    int u = idx % Wout;
    int v = (idx / Wout) % H;
    int b = idx / (Wout * H);
    const unsigned int* src = zbuf + ((size_t)b * H + v) * W;
    int a0 = src0[u];
    int a1 = src1[u];
    unsigned int k0 = src[a0];
    float z0 = (k0 & 0x80000000u)
        ? __uint_as_float(k0 & 0x7FFFFFFFu)
        : __uint_as_float(~k0);
    float acc = w0[u] * ((z0 < 1e29f) ? (z0 - gel_top_z) * 1000.0f : 0.0f);
    if (a1 >= 0) {
        unsigned int k1 = src[a1];
        float z1 = (k1 & 0x80000000u)
            ? __uint_as_float(k1 & 0x7FFFFFFFu)
            : __uint_as_float(~k1);
        acc += w1[u] * ((z1 < 1e29f) ? (z1 - gel_top_z) * 1000.0f : 0.0f);
    }
    out[idx] = acc < 0.0f ? acc : 0.0f;
}
"""


@lru_cache(maxsize=4)
def _width_interp_tables(src_w: int | None = None, dst_w: int | None = None):
    """Per-output-column (src0, src1, w0, w1) from cv2 INTER_LINEAR weights."""
    src_w = RENDER_W if src_w is None else int(src_w)
    dst_w = DEPTH_W if dst_w is None else int(dst_w)
    wx = _width_bilinear_weights(src_w, dst_w)
    src0 = np.full(dst_w, 0, dtype=np.int32)
    src1 = np.full(dst_w, -1, dtype=np.int32)
    w0 = np.zeros(dst_w, dtype=np.float32)
    w1 = np.zeros(dst_w, dtype=np.float32)
    for u in range(dst_w):
        nz = np.nonzero(wx[:, u])[0]
        if nz.size == 0:
            src0[u] = 0
            w0[u] = 1.0
        else:
            src0[u] = int(nz[0])
            w0[u] = float(wx[nz[0], u])
            if nz.size > 1:
                src1[u] = int(nz[1])
                w1[u] = float(wx[nz[1], u])
    return src0, src1, w0, w1


def _ray_triangle_depth_gpu_raster_keys(base3_g, R_g, t_g):
    """Rasterize only; return the raw encoded z keys (B, H, W) uint32."""
    import cupy as cp

    B = R_g.shape[0]
    N = base3_g.shape[0]
    W, H = RENDER_W, RENDER_H
    zbuf = cp.full((B, W * H), np.uint32(0xFFFFFFFF), dtype=cp.uint32)
    k = cp.RawKernel(_RASTER_KERNELS, "raster_depth_kernel")
    block = 256
    grid = ((N + block - 1) // block, B)
    k(
        grid,
        (block,),
        (
            base3_g.reshape(N, 9),
            R_g,
            t_g,
            zbuf,
            N,
            W,
            H,
            np.float32(CAM_Z),
            np.float32(CAMERA_TAN_H),
            np.float32(CAMERA_TAN_V),
            np.float32(CAMERA_PRINCIPAL_U),
            np.float32(CAMERA_PRINCIPAL_V),
            np.float32(CAMERA_NEAR_M),
        ),
    )
    return zbuf.reshape(B, H, W)


_WIDTH_WEIGHTS_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _width_bilinear_weights(src_w: int, dst_w: int):
    """cv2 INTER_LINEAR-equivalent width-only weights (src_w, dst_w)."""
    key = (src_w, dst_w)
    cached = _WIDTH_WEIGHTS_CACHE.get(key)
    if cached is None:
        import cv2

        w = np.eye(src_w, dtype=np.float32)
        out = cv2.resize(w, (dst_w, src_w), interpolation=cv2.INTER_LINEAR)
        cached = np.asarray(out, dtype=np.float32)
        _WIDTH_WEIGHTS_CACHE[key] = cached
    return cached


def mesh_depth_from_poses(base3_g, R_g, t_g):
    """Indentation for per-env poses (B,DEPTH_H,DEPTH_W) float32 mm, <= 0."""
    import cupy as cp

    zbuf = _ray_triangle_depth_gpu_raster_keys(base3_g, R_g, t_g)
    src0, src1, w0, w1 = _width_interp_tables()
    out = cp.empty((zbuf.shape[0], DEPTH_H, DEPTH_W), dtype=cp.float32)
    k = cp.RawKernel(_FINISH_KERNELS, "finish_depth_kernel")
    total = out.size
    block = 256
    k(
        ((total + block - 1) // block,),
        (block,),
        (
            zbuf,
            cp.asarray(src0, dtype=cp.int32),
            cp.asarray(src1, dtype=cp.int32),
            cp.asarray(w0, dtype=cp.float32),
            cp.asarray(w1, dtype=cp.float32),
            out,
            RENDER_W,
            DEPTH_H,
            DEPTH_W,
            zbuf.shape[0],
            np.float32(CAMERA_TOP_Z_M),
        ),
    )
    return out


def depth_from_poses(
    base3_g,
    R,
    t,
    chunk: int = 0,
):
    """Indentation for per-env poses: base (N,3,3) world tris at rest.

    R: (B,3,3) float32 rotations; t: (B,3) float32 translations (m) applied
    on top of the rest pose.  Returns (B,DEPTH_H,DEPTH_W) float32 indentation mm,
    <= 0 (chunked to bound per-launch memory).
    """
    import cupy as cp

    B = R.shape[0]
    chunk = chunk or B
    R_g = cp.asarray(R, dtype=cp.float32)
    t_g = cp.asarray(t, dtype=cp.float32)
    out = cp.empty((B, DEPTH_H, DEPTH_W), dtype=cp.float32)
    for k in range(0, B, chunk):
        sl = slice(k, k + chunk)
        out[sl] = mesh_depth_from_poses(base3_g, R_g[sl], t_g[sl])
    return out


class DepthRenderer:
    """Preallocated HydroShear depth workspace for a fixed batch size."""

    def __init__(self, max_batch: int, *, chunk_size: int = 512) -> None:
        import cupy as cp

        if max_batch <= 0 or chunk_size <= 0:
            raise ValueError("depth renderer batch and chunk sizes must be positive")
        self.max_batch = int(max_batch)
        self.chunk_size = min(int(chunk_size), self.max_batch)
        self._zbuf = cp.empty((self.chunk_size, RENDER_H, RENDER_W), dtype=cp.uint32)
        self._depth = cp.empty((self.chunk_size, DEPTH_H, DEPTH_W), dtype=cp.float32)
        tables = _width_interp_tables()
        self._src0, self._src1, self._w0, self._w1 = (cp.asarray(value) for value in tables)
        self._raster = cp.RawKernel(_RASTER_KERNELS, "raster_depth_kernel")
        self._finish = cp.RawKernel(_FINISH_KERNELS, "finish_depth_kernel")
        self._cp = cp

    def render_into(self, triangles: Any, rotations: Any, translations: Any, output: Any) -> None:
        """Render all rows into an existing ``(B, H, W)`` output array."""
        batch = int(rotations.shape[0])
        if batch <= 0 or batch > self.max_batch:
            raise ValueError(f"depth batch must be in [1, {self.max_batch}]")
        if translations.shape != (batch, 3):
            raise ValueError("depth translations must have shape (batch, 3)")
        if rotations.shape != (batch, 3, 3):
            raise ValueError("depth rotations must have shape (batch, 3, 3)")
        if output.shape != (batch, DEPTH_H, DEPTH_W):
            raise ValueError("depth output shape differs from the configured batch")

        cp = self._cp
        for start in range(0, batch, self.chunk_size):
            stop = min(start + self.chunk_size, batch)
            count = stop - start
            zbuf = self._zbuf[:count].reshape(count, RENDER_H * RENDER_W)
            zbuf.fill(np.uint32(0xFFFFFFFF))
            base = cp.asarray(triangles, dtype=cp.float32).reshape(-1, 9)
            rot = rotations[start:stop]
            trans = translations[start:stop]
            triangle_count = int(base.shape[0])
            block = 256
            self._raster(
                ((triangle_count + block - 1) // block, count),
                (block,),
                (
                    base,
                    rot.reshape(count, 9),
                    trans,
                    zbuf,
                    triangle_count,
                    RENDER_W,
                    RENDER_H,
                    np.float32(CAM_Z),
                    np.float32(CAMERA_TAN_H),
                    np.float32(CAMERA_TAN_V),
                    np.float32(CAMERA_PRINCIPAL_U),
                    np.float32(CAMERA_PRINCIPAL_V),
                    np.float32(CAMERA_NEAR_M),
                ),
            )
            total = count * DEPTH_H * DEPTH_W
            self._finish(
                ((total + block - 1) // block,),
                (block,),
                (
                    zbuf,
                    self._src0,
                    self._src1,
                    self._w0,
                    self._w1,
                    self._depth[:count],
                    RENDER_W,
                    DEPTH_H,
                    DEPTH_W,
                    count,
                    np.float32(CAMERA_TOP_Z_M),
                ),
            )
            output[start:stop] = self._depth[:count]
