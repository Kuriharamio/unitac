"""HydroShear marker-shear matrix kernels.

The validated GEMM implementation is derived from HydroShear by MMintLab (MIT);
see ``licenses/HydroShear-MIT.txt``.

Matmul (GEMM) reformulation of the HydroShear marker-shear kernel.

Despite the ``fft_`` sibling module name, shear is *not* an FFT case: the sum runs
over irregularly-placed indenter points, so ``||p_i - a_j||^2`` is not a function
of an index offset and there is no convolution structure (see FFT.md, section 2).

What shear *is* is a matrix multiply. The reference computation
(``HydroSoftSensor marker shear``) is::

    M[i, c] = -shear_scale * sum_j  h_j * fbar[j, c]
                                    * exp(-lambda_s * || p_i - a_j ||^2)

where a_j = indenter_pts[j] + fbar[j] and h_j = fbar[j, normal_axis]. Splitting the
summand into a part that depends on (i, j) and a part that depends only on j:

    W[i, j] = exp(-lambda_s * || p_i - a_j ||^2)      (E, N, P)
    Q[j, c] = h_j * fbar[j, c]                        (E, P, 3)
    M       = -shear_scale * (W @ Q)                  (E, N, 3)

This is algebraically identical to the dense sum for *any* point layout (no grid
assumption), and never materialises the dense (E, P, N, 3) pairwise tensor that
the reference builds -- which is the real memory bottleneck since P >> N. It is
both faster and far lighter on memory, and it completes cases where the dense path
runs out of GPU memory.

Dependency-free (torch only) so it can be unit-tested without isaacgym / open3d.
"""

import torch


def _as_env_param(value, num_envs, device, dtype):
    """Broadcast a scalar / (E,) / (E,1) coefficient to (E, 1, 1)."""
    if not torch.is_tensor(value):
        value = torch.tensor(value, device=device, dtype=dtype)
    value = value.to(device=device, dtype=dtype).reshape(-1)
    if value.numel() == 1:
        value = value.expand(num_envs)
    assert value.numel() == num_envs, (
        f"coefficient has {value.numel()} entries, expected 1 or num_envs={num_envs}"
    )
    return value.reshape(num_envs, 1, 1)


def shear_dense(
    tactile_pts, indenter_pts, fbar, lambda_s, shear_scale, normal_axis=2, reverse_z=True
):
    """Reference dense O(P*N) shear, mirroring ``get_marker_shear`` (given fbar).

    Args:
        tactile_pts:  (E, N, 3) tactile points in the elastomer frame.
        indenter_pts: (E, P, 3) indenter points in the elastomer frame.
        fbar:         (E, P, 3) hydrosoft forces at the indenter points.
        lambda_s:     scalar / (E,) / (E,1) Gaussian decay.
        shear_scale:  scalar / (E,) / (E,1) output scale.
        normal_axis:  which component of fbar is the "height" h_j.
        reverse_z:    sign convention flag, matching the reference.

    Returns:
        (E, N, 3) marker shear displacement.
    """
    E, P, _ = indenter_pts.shape
    N = tactile_pts.shape[1]
    lam = _as_env_param(lambda_s, E, tactile_pts.device, tactile_pts.dtype)
    scale = _as_env_param(shear_scale, E, tactile_pts.device, tactile_pts.dtype)

    a = indenter_pts + fbar  # (E, P, 3)
    h = fbar[:, :, normal_axis]  # (E, P)
    tp = tactile_pts.unsqueeze(1).expand(E, P, N, 3)
    ap = a.unsqueeze(2).expand(E, P, N, 3)
    fb = fbar.unsqueeze(2).expand(E, P, N, 3)
    norm = torch.norm(tp - ap, dim=-1)  # (E, P, N)
    hh = h.unsqueeze(-1).unsqueeze(-1).expand(E, P, N, 3)
    ge = torch.exp(-lam * norm**2).unsqueeze(-1).expand(E, P, N, 3)
    sign = -1.0 * (-1 if reverse_z else 1)
    M = torch.sum(scale.unsqueeze(-1) * hh * fb * ge * sign, dim=1)  # (E, N, 3)
    return M


def shear_matmul(
    tactile_pts, indenter_pts, fbar, lambda_s, shear_scale, normal_axis=2, reverse_z=True
):
    """GEMM shear: exact, memory-light replacement for ``shear_dense``.

    Same arguments and return shape as ``shear_dense``. Never forms the
    (E, P, N, 3) intermediate; peak extra memory is the (E, N, P) weight matrix.
    """
    E = indenter_pts.shape[0]
    lam = _as_env_param(lambda_s, E, tactile_pts.device, tactile_pts.dtype)  # (E,1,1)
    scale = _as_env_param(shear_scale, E, tactile_pts.device, tactile_pts.dtype).reshape(E, 1, 1)

    a = indenter_pts + fbar  # (E, P, 3)
    h = fbar[:, :, normal_axis]  # (E, P)
    d2 = torch.cdist(tactile_pts, a).pow(2)  # (E, N, P)
    w = torch.exp(-lam * d2)  # (E, N, P)
    q = h.unsqueeze(-1) * fbar  # (E, P, 3)
    sign = -1.0 * (-1 if reverse_z else 1)
    M = scale * sign * torch.bmm(w, q)  # (E, N, 3)
    return M


def shear_matmul_active(
    tactile_pts,
    indenter_pts,
    fbar,
    lambda_s,
    shear_scale,
    max_points,
    normal_axis=2,
    reverse_z=True,
):
    """GEMM shear over the non-zero contact points only.

    HydroFOTS produces exactly zero force for sampled points outside contact.
    Selecting those points before the Gaussian kernel reduces the ``(E, N, P)``
    distance matrix to ``(E, N, K)`` while preserving the dense result when
    every environment has at most ``K`` active points.  If an environment
    exceeds the cap, the function falls back to the uncropped GEMM path so the
    optimization cannot silently change the sensor output.
    """
    if max_points is None:
        return shear_matmul(
            tactile_pts,
            indenter_pts,
            fbar,
            lambda_s,
            shear_scale,
            normal_axis=normal_axis,
            reverse_z=reverse_z,
        )

    max_points = int(max_points)
    if max_points <= 0 or max_points >= indenter_pts.shape[1]:
        return shear_matmul(
            tactile_pts,
            indenter_pts,
            fbar,
            lambda_s,
            shear_scale,
            normal_axis=normal_axis,
            reverse_z=reverse_z,
        )

    active = fbar.abs().amax(dim=-1).gt(0)
    active_count = active.sum(dim=-1)
    # This is an exceptional safety branch.  It deliberately falls back to
    # the exact full-point calculation instead of truncating contact forces.
    # CUDA uses the exact dense path directly: reading ``active_count`` into
    # Python would synchronize the device on every frame.
    if active_count.device.type == "cuda":
        return shear_matmul(
            tactile_pts,
            indenter_pts,
            fbar,
            lambda_s,
            shear_scale,
            normal_axis=normal_axis,
            reverse_z=reverse_z,
        )
    if bool((active_count > max_points).any()):
        return shear_matmul(
            tactile_pts,
            indenter_pts,
            fbar,
            lambda_s,
            shear_scale,
            normal_axis=normal_axis,
            reverse_z=reverse_z,
        )

    _, indices = torch.topk(active.to(dtype=torch.int32), max_points, dim=1, sorted=False)
    gather_indices = indices.unsqueeze(-1).expand(-1, -1, 3)
    compact_indenter = torch.gather(indenter_pts, 1, gather_indices)
    compact_fbar = torch.gather(fbar, 1, gather_indices)
    return shear_matmul(
        tactile_pts,
        compact_indenter,
        compact_fbar,
        lambda_s,
        shear_scale,
        normal_axis=normal_axis,
        reverse_z=reverse_z,
    )


def force_dense(tactile_pts, indenter_pts, fbar, lambda_s):
    """Reference dense force-field interpolation at tactile points.

    The HydroFOTS solver produces ``fbar`` at the sampled indenter points.
    This maps those vectors to the marker points with the same affected-point
    Gaussian kernel used by the shear path, but without the displacement
    scale/normal-height factors.  The result is a marker force vector field
    with shape ``(E, N, 3)``.
    """
    E, P, _ = indenter_pts.shape
    lam = _as_env_param(lambda_s, E, tactile_pts.device, tactile_pts.dtype)
    affected = indenter_pts + fbar
    d2 = torch.cdist(tactile_pts, affected).pow(2)  # (E, N, P)
    weights = torch.exp(-lam * d2)
    return (weights.unsqueeze(-1) * fbar.unsqueeze(1)).sum(dim=2)


def force_matmul(tactile_pts, indenter_pts, fbar, lambda_s):
    """Memory-light marker force-field interpolation.

    This is the GEMM form of :func:`force_dense`; it returns the same weighted
    marker force vectors without materializing the ``(E, P, N, 3)`` tensor.
    """
    E = indenter_pts.shape[0]
    lam = _as_env_param(lambda_s, E, tactile_pts.device, tactile_pts.dtype)
    affected = indenter_pts + fbar
    d2 = torch.cdist(tactile_pts, affected).pow(2)  # (E, N, P)
    weights = torch.exp(-lam * d2)
    return torch.bmm(weights, fbar)
