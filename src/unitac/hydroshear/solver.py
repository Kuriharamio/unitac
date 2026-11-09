"""HydroSoft history and marker-field solver.

Derived from HydroShear by MMintLab (MIT); see ``licenses/HydroShear-MIT.txt``.
"""

from typing import Any

import torch
import torch.nn.functional as F


class HydroSoftSensor:
    """
    TactileFieldSensor gets sdf of indenter and queries elastomer points.
    HydroSoftSensor is the reverse where it gets sdf of elastomer and queries indenter points.
    """

    def __init__(
        self,
        mu=0.5,
        lambda_d=20_000.0,
        lambda_s=10_800.0,
        shear_scale=1000 / 0.065,
        dilate_scale=1000 / 0.065,
        normal_axis=2,
        K=1.0,
        E=1.0,
        A=1.0,
        reverse_z=False,
        use_fft_dilation=False,
        fft_dilation_min_pts=256,
        fft_dilation_grid_shape=(9, 7),
        use_matmul_shear=False,
        device: Any = "cuda",
    ):
        super().__init__()
        self.device = device

        # initialize coefficients
        self.K = K
        self.E = E
        self.A = A
        self.mu = mu

        self.lambda_s = lambda_s
        self.lambda_d = lambda_d

        self.shear_scale = shear_scale
        self.dilate_scale = dilate_scale
        for name, value in (
            ("reverse_z", reverse_z),
            ("use_fft_dilation", use_fft_dilation),
            ("use_matmul_shear", use_matmul_shear),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be bool")
        if (
            isinstance(fft_dilation_min_pts, bool)
            or not isinstance(fft_dilation_min_pts, int)
            or fft_dilation_min_pts <= 0
        ):
            raise ValueError("fft_dilation_min_pts must be a positive integer")
        if (
            isinstance(fft_dilation_grid_shape, (str, bytes))
            or len(fft_dilation_grid_shape) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in fft_dilation_grid_shape
            )
        ):
            raise ValueError("fft_dilation_grid_shape must contain two positive integers")
        self.reverse_z = reverse_z
        self.use_fft_dilation = use_fft_dilation
        self.fft_dilation_min_pts = fft_dilation_min_pts
        self.fft_dilation_grid_shape = tuple(fft_dilation_grid_shape)
        self._fft_dilation = None
        self.use_matmul_shear = use_matmul_shear

        self.hydrosoft_forces: Any = None
        self.prev_sdf: Any = None
        self.prev_grasped_obj_pts_elastomer: Any = None

        if isinstance(normal_axis, bool) or not isinstance(normal_axis, int) or normal_axis != 2:
            raise ValueError("HydroSoftSensor currently supports only normal_axis=2")
        self.normal_axis = normal_axis
        self.tangent_axes = [i for i in range(3) if i != normal_axis]

    def initialize(self, num_envs, num_obj_pts):
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        if isinstance(num_obj_pts, bool) or not isinstance(num_obj_pts, int) or num_obj_pts <= 0:
            raise ValueError("num_obj_pts must be a positive integer")
        self.num_envs = int(num_envs)
        self.num_obj_pts = int(num_obj_pts)
        self.hydrosoft_forces = torch.zeros(
            (self.num_envs, self.num_obj_pts, 3), device=self.device
        )
        self.prev_sdf = torch.empty(
            (self.num_envs, self.num_obj_pts), device=self.device, dtype=torch.float32
        )
        self.prev_grasped_obj_pts_elastomer = torch.empty(
            (self.num_envs, self.num_obj_pts, 3),
            device=self.device,
            dtype=torch.float32,
        )
        self._reset_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        # Static solver coefficients and the Heaviside fill value are staged
        # once, outside the steady-state device path.
        self._lambda_s_device = self._device_coefficient(self.lambda_s)
        self._lambda_d_device = self._device_coefficient(self.lambda_d)
        self._shear_scale_device = self._device_coefficient(self.shear_scale)
        self._dilate_scale_device = self._device_coefficient(self.dilate_scale)
        self._mu_device = self._device_coefficient(self.mu)
        self._normal_stiffness_device = self._device_coefficient(self.E) * self._device_coefficient(
            self.A
        )
        self._tangent_stiffness_device = self._device_coefficient(
            self.K
        ) * self._device_coefficient(self.A)
        self._heaviside_zero = torch.zeros(1, device=self.device)

    def _device_coefficient(self, value):
        coefficient = torch.as_tensor(value, device=self.device)
        if coefficient.dtype == torch.bool or coefficient.is_complex():
            raise TypeError("HydroSoft coefficients must be real numeric values")
        coefficient = coefficient.to(dtype=torch.float32)
        if coefficient.ndim not in (0, 1) or coefficient.numel() == 0:
            raise ValueError("HydroSoft coefficients must be scalars or one-dimensional arrays")
        if not bool(torch.isfinite(coefficient).all()) or bool(torch.any(coefficient < 0)):
            raise ValueError("HydroSoft coefficients must be finite and non-negative")
        if coefficient.numel() == 1:
            return coefficient.reshape(())
        if coefficient.numel() != self.num_envs:
            raise ValueError(f"HydroSoft coefficient must be scalar or have {self.num_envs} values")
        return coefficient.reshape(self.num_envs)

    def _coefficient_for_rank(self, value, rank):
        if value.numel() == 1:
            return value.reshape((1,) * rank)
        return value.reshape((self.num_envs,) + (1,) * (rank - 1))

    def reset(self, env_ids=None):
        """Reset temporal state for all or a subset of batch environments."""
        if self.hydrosoft_forces is None:
            raise RuntimeError("HydroSoftSensor must be initialized before reset")
        if env_ids is None:
            self._reset_mask.fill_(True)
            self.hydrosoft_forces.zero_()
            return
        if isinstance(env_ids, torch.Tensor):
            selector = env_ids.detach()
            if selector.device.type == "cuda" and selector.device != self.hydrosoft_forces.device:
                raise ValueError("HydroSoft reset selector must use the solver device")
            if selector.device.type not in ("cpu", "cuda"):
                raise TypeError("HydroSoft reset selector must use CPU or solver-device CUDA")
        else:
            selector = torch.as_tensor(env_ids)
        if selector.ndim != 1:
            raise ValueError("HydroSoft reset selector must be one-dimensional")
        if selector.numel() == 0:
            return
        if selector.dtype == torch.bool:
            if selector.shape != (self.num_envs,):
                raise ValueError(
                    f"HydroSoftSensor boolean reset mask must have length {self.num_envs}"
                )
            selector = selector.to(device=self.device)
            self._reset_mask.logical_or_(selector)
            mask = selector[:, None, None]
            self.hydrosoft_forces.masked_fill_(mask, 0.0)
            return
        if selector.dtype not in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise TypeError("HydroSoft reset selector must contain integers or booleans")
        ids = selector.to(dtype=torch.long)
        if ids.device != self.hydrosoft_forces.device:
            ids = ids.to(device=self.hydrosoft_forces.device)
        valid = torch.all((ids >= 0) & (ids < self.num_envs))
        sorted_ids = torch.sort(ids).values
        distinct = torch.all(sorted_ids[1:] != sorted_ids[:-1])
        if ids.device.type == "cuda":
            _assert_async(valid, "HydroSoftSensor reset ids are out of range")
            _assert_async(distinct, "HydroSoft reset ids must be distinct")
        else:
            if not bool(valid):
                raise IndexError("HydroSoftSensor reset ids are out of range")
            if not bool(distinct):
                raise IndexError("HydroSoft reset ids must be distinct")
        self._reset_mask.index_fill_(0, ids, True)
        self.hydrosoft_forces.index_fill_(0, ids, 0.0)

    def step_hydrosoft_forces(self, grasped_obj_pts_in_elastomer, grasped_obj_sdf, reverse_z=None):
        if reverse_z is None:
            reverse_z = self.reverse_z
        sdf = grasped_obj_sdf

        # A sparse environment reset must not inherit the previous contact
        # state from another episode.  Seed reset rows with the current frame
        # before calculating the temporal displacement.
        reset_mask = self._reset_mask
        reset_points = reset_mask.reshape((-1,) + (1,) * (grasped_obj_pts_in_elastomer.ndim - 1))
        reset_sdf = reset_mask.reshape((-1,) + (1,) * (sdf.ndim - 1))
        torch.where(
            reset_points,
            grasped_obj_pts_in_elastomer,
            self.prev_grasped_obj_pts_elastomer,
            out=self.prev_grasped_obj_pts_elastomer,
        )
        torch.where(reset_sdf, sdf, self.prev_sdf, out=self.prev_sdf)
        self._reset_mask.zero_()

        # alpha_d = (F.relu(-sdf) - F.relu(-self.prev_sdf)) / (self.prev_sdf - sdf + 1e-8)
        # displacement = alpha_d.unsqueeze(-1) * (self.prev_grasped_obj_pts_elastomer - grasped_obj_pts_in_elastomer) # (num_envs, num_pts, 3)
        alpha_d = -torch.divide(F.relu(-self.prev_sdf) - F.relu(-sdf), self.prev_sdf - sdf)
        alpha_d = torch.where(torch.isnan(alpha_d), -0.5 * (torch.sign(sdf) - 1), alpha_d)
        displacement = alpha_d.unsqueeze(-1) * (
            self.prev_grasped_obj_pts_elastomer - grasped_obj_pts_in_elastomer
        )

        # NOTE: assume +z is out of elastomer
        # self.hydrosoft_forces[:,:,2] += self.E * self.A * displacement[:, :, 2]
        # self.hydrosoft_forces[:,:,:2] += self.K * self.A * displacement[:, :, :2]
        # fn = self.hydrosoft_forces[:,:,2]
        # ft = self.hydrosoft_forces[:,:,:2]
        fn = (
            self.hydrosoft_forces[:, :, self.normal_axis]
            + self._coefficient_for_rank(self._normal_stiffness_device, 2)
            * displacement[:, :, self.normal_axis]
        )
        ft = (
            self.hydrosoft_forces[:, :, self.tangent_axes]
            + self._coefficient_for_rank(self._tangent_stiffness_device, 3)
            * displacement[:, :, self.tangent_axes]
        )

        # fn_bar = F.relu(fn)
        fn_bar = F.relu(fn * (-1 if reverse_z else 1))

        norm_ft = torch.norm(ft, dim=-1)  # (num_envs, num_pts)
        # norm_ft[norm_ft < 1e-8] = 1
        # ft_bar = torch.zeros_like(ft)
        # ft_bar[norm_ft > 1e-8] = torch.clamp(self.mu * fn_bar[norm_ft > 1e-8] / norm_ft[norm_ft > 1e-8], 1).unsqueeze(-1) * ft[norm_ft > 1e-8]
        # ft_bar = torch.minimum(self.mu * fn_bar / (norm_ft + 1e-8), torch.ones_like(norm_ft)).unsqueeze(-1) * ft
        ft_bar = (
            torch.minimum(
                self._coefficient_for_rank(self._mu_device, 2) * fn_bar, norm_ft
            ).unsqueeze(-1)
            * ft
            / (norm_ft.unsqueeze(-1) + 1e-8)
        )

        fbar = torch.cat([ft_bar, fn_bar.unsqueeze(-1) * (-1 if reverse_z else 1)], dim=-1)
        fbar = (
            torch.heaviside(-sdf, values=self._heaviside_zero).unsqueeze(-1) * fbar
        )  # zero out forces if not in contact

        self.hydrosoft_forces.copy_(fbar)

        # Fixed history buffers outlive the reused geometry and SDF workspaces.
        self.prev_grasped_obj_pts_elastomer.copy_(grasped_obj_pts_in_elastomer)
        self.prev_sdf.copy_(sdf)

        return fbar

    def get_dilation_displacement(self, tactile_pts_elastomer, tactile_pts_height):
        """
        tactile_pts_elastomer: (num_envs, num_tactile_pts, 3)
        tactile_pts_height: (num_envs, num_tactile_pts)
        """
        # sum_i delta_h_i (M - C_i) * exp(-lambda_d * \|M - C_i\|^2)
        num_envs, num_tactile_pts, _ = tactile_pts_elastomer.shape
        if (
            self.use_fft_dilation
            and self._fft_dilation is None
            and num_tactile_pts >= self.fft_dilation_min_pts
            and self.fft_dilation_grid_shape[0] * self.fft_dilation_grid_shape[1] == num_tactile_pts
        ):
            from .fft_dilation import DilationFFT, infer_grid_geometry

            info = infer_grid_geometry(tactile_pts_elastomer[0], self.fft_dilation_grid_shape)
            if info["is_planar"]:
                self._fft_dilation = DilationFFT(
                    self.fft_dilation_grid_shape,
                    info["spacing"],
                    tangent_axes=info["tangent_axes"],
                    device=tactile_pts_elastomer.device,
                    dtype=tactile_pts_elastomer.dtype,
                )
        if self._fft_dilation is not None:
            return self._fft_dilation(
                tactile_pts_height, self._lambda_d_device, self._dilate_scale_device
            )

        dvec = tactile_pts_elastomer.unsqueeze(2) - tactile_pts_elastomer.unsqueeze(
            1
        )  # (num_envs, num_tactile_pts, num_tactile_pts, 3)
        norm_dvec = torch.norm(dvec, dim=-1)  # (num_envs, num_tactile_pts, num_tactile_pts)

        lambda_d = self._coefficient_for_rank(self._lambda_d_device, 3)
        dilate_scale = self._coefficient_for_rank(self._dilate_scale_device, 4)
        gaussian_exp = (
            torch.exp(-lambda_d * norm_dvec**2)
            .unsqueeze(-1)
            .expand(num_envs, num_tactile_pts, num_tactile_pts, 3)
        )  # (num_envs, num_tactile_pts, num_tactile_pts, 3)
        h = (
            tactile_pts_height.unsqueeze(1).unsqueeze(-1).expand(num_envs, 1, num_tactile_pts, 3)
        )  # (num_envs, num_tactile_pts, num_tactile_pts, 3)
        Mdilate = (dilate_scale * h * dvec * gaussian_exp).sum(dim=2)

        return Mdilate

    def get_marker_displacement_from_force(
        self,
        tactile_pts_elastomer,
        tactile_pts_height,
        obj_pts_elastomer,
        fbar,
        reverse_z=False,
    ):
        """Propagate an already-computed contact force to gel markers.

        The contact update is intentionally separate from this method: a
        caller can compute ``fbar`` once per physics step and then request the
        marker displacement without advancing the sensor's temporal state a
        second time.
        """
        Mdilate, Mshear = self.get_marker_displacement_components_from_force(
            tactile_pts_elastomer,
            tactile_pts_height,
            obj_pts_elastomer,
            fbar,
            reverse_z=reverse_z,
        )
        return Mdilate + Mshear

    def get_marker_displacement_components_from_force(
        self,
        tactile_pts_elastomer,
        tactile_pts_height,
        obj_pts_elastomer,
        fbar,
        reverse_z=False,
    ):
        """Return dilation and shear marker displacement components.

        Both components are evaluated in one pass before they are combined
        into the total marker displacement.
        """
        Mdilate = self.get_dilation_displacement(tactile_pts_elastomer, tactile_pts_height)
        from .fft_shear import shear_dense

        if self.use_matmul_shear:
            from .fft_shear import shear_matmul_active

            Mshear = shear_matmul_active(
                tactile_pts_elastomer,
                obj_pts_elastomer,
                fbar,
                self._lambda_s_device,
                self._shear_scale_device,
                max_points=512,
                normal_axis=self.normal_axis,
                reverse_z=reverse_z,
            )
        else:
            Mshear = shear_dense(
                tactile_pts_elastomer,
                obj_pts_elastomer,
                fbar,
                self._lambda_s_device,
                self._shear_scale_device,
                normal_axis=self.normal_axis,
                reverse_z=reverse_z,
            )
        return Mdilate, Mshear

    def step_marker_displacement(
        self,
        tactile_pts_elastomer,
        tactile_pts_height,
        obj_pts_elastomer,
        obj_pts_sdf,
        reverse_z=False,
    ):
        """Advance one frame and return force plus total marker displacement."""
        fbar = self.step_hydrosoft_forces(obj_pts_elastomer, obj_pts_sdf, reverse_z=reverse_z)
        Mdilate, Mshear = self.get_marker_displacement_components_from_force(
            tactile_pts_elastomer,
            tactile_pts_height,
            obj_pts_elastomer,
            fbar,
            reverse_z=reverse_z,
        )
        marker_displacement = Mdilate + Mshear
        return fbar, marker_displacement


def _assert_async(condition: torch.Tensor, message: str) -> None:
    assertion = getattr(torch, "_assert_async", None)
    if not callable(assertion):
        raise RuntimeError("Torch build does not provide asynchronous CUDA assertions")
    assertion(condition, message)
