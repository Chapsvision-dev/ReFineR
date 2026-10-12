from typing import Optional, List, Tuple, Literal

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

from refiner.utils.lie import SO3_exp
from refiner.camera.ready_camera import ready_camera


def square_err(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    fx: float,
    fy: float,
    r_axis: np.ndarray,
    t_axis: np.ndarray,
) -> List[np.ndarray]:
    o = np.ones_like(x)

    rx = np.mean((x**2 * y**2 + (fy**2 + y**2) ** 2) / (fy**2) * r_axis[0] * r_axis[0])
    ry = np.mean((x**2 * y**2 + (fx**2 + x**2) ** 2) / (fx**2) * r_axis[1] * r_axis[1])
    rz = np.mean((fx**2 / fy**2 * y**2 + fy**2 / fx**2 * x**2) * r_axis[2] * r_axis[2])

    tx = np.mean(fx**2 / z**2 * t_axis[0] * t_axis[0])
    ty = np.mean(fy**2 / z**2 * t_axis[1] * t_axis[1])
    tz = np.mean((x**2 / z**2 + y**2 / z**2) * t_axis[2] * t_axis[2])

    rxry = np.mean(-x * y / (fx * fy) * (fx**2 + fy**2 + x**2 + y**2) * r_axis[0] * r_axis[1])
    rxrz = np.mean((x * y**2 * (fx / fy**2 - 1 / fx) - fy**2 / fx * x) * r_axis[0] * r_axis[2])
    ryrz = np.mean((x**2 * y * (fy / fx**2 - 1 / fy) - fx**2 / fy * y) * r_axis[1] * r_axis[2])

    rxtx = np.mean(-fx / fy * x * y / z * r_axis[0] * t_axis[0])
    ryty = np.mean(fy / fx * x * y / z * r_axis[1] * t_axis[1])

    rxty = np.mean(-(fy**2 + y**2) / z * r_axis[0] * t_axis[1])
    rytx = np.mean((fx**2 + x**2) / z * r_axis[1] * t_axis[0])

    rxtz = np.mean((x**2 * y + y * fy**2 + y**3) / (fy * z) * r_axis[0] * t_axis[2])
    rytz = np.mean(-(x * y**2 + x * fx**2 + x**3) / (fx * z) * r_axis[1] * t_axis[2])

    rztx = np.mean(-(fx**2) / fy * y / z * r_axis[2] * t_axis[0])
    rzty = np.mean(fy**2 / fx * x / z * r_axis[2] * t_axis[1])

    rztz = np.mean(x * y / z * (fx / fy - fy / fx) * r_axis[2] * t_axis[2])

    txty = 0

    txtz = np.mean(-fx * x / z**2 * t_axis[0] * t_axis[2])
    tytz = np.mean(-fy * y / z**2 * t_axis[1] * t_axis[2])

    return [
        rx,
        ry,
        rz,
        tx,
        ty,
        tz,
        rxry,
        rxrz,
        rxtx,
        rxty,
        rxtz,
        ryrz,
        rytx,
        ryty,
        rytz,
        rztx,
        rzty,
        rztz,
        txty,
        txtz,
        tytz,
    ]


def big_cov_err(errs: List[np.ndarray]) -> np.ndarray:
    [
        rx,
        ry,
        rz,
        tx,
        ty,
        tz,
        rxry,
        rxrz,
        rxtx,
        rxty,
        rxtz,
        ryrz,
        rytx,
        ryty,
        rytz,
        rztx,
        rzty,
        rztz,
        txty,
        txtz,
        tytz,
    ] = errs

    # fmt: off
    S = np.array(
        [
            [  rx, rxry, rxrz, rxtx, rxty, rxtz],
            [rxry,   ry, ryrz, rytx, ryty, rytz],
            [rxrz, ryrz,   rz, rztx, rzty, rztz],
            [rxtx, rytx, rztx,   tx, txty, txtz],
            [rxty, ryty, rzty, txty,   ty, tytz],
            [rxtz, rytz, rztz, txtz, tytz,   tz],
        ]
    )
    # fmt: on

    return S


def random_rotation_matrix(theta, rng=None, nsamples=1):
    axis = (
        np.random.rand(nsamples, 3) - 0.5 if rng is None else rng.random((nsamples, 3))
    )  # N x 3
    axis /= np.linalg.norm(axis, axis=1, keepdims=True) + 1e-8
    return R.from_rotvec(axis * theta).as_matrix()  # N x 3 x 3


def random_pose_rot_transl(
    t: float = 0,
    r: float = 0,
    t_axis: Optional[List[float]] = None,
    r_axis: Optional[List[float]] = None,
    rng: Optional[np.random.Generator] = None,
    nsamples: int = 1,
):
    if r_axis is None:
        r_axis = (
            np.random.rand(nsamples, 3) - 0.5
            if rng is None
            else rng.random((nsamples, 3))
        )  # N x 3
    else:
        r_axis = np.repeat(np.expand_dims(r_axis, axis=0), nsamples, axis=0).astype(
            np.float32
        )
    if t_axis is None:
        t_axis = (
            np.random.rand(nsamples, 3) - 0.5
            if rng is None
            else rng.random((nsamples, 3))
        )  # N x 3
    else:
        t_axis = np.repeat(np.expand_dims(t_axis, axis=0), nsamples, axis=0).astype(
            np.float32
        )
    r_axis /= np.linalg.norm(r_axis, axis=1, keepdims=True) + 1e-8
    t_axis /= np.linalg.norm(t_axis, axis=1, keepdims=True) + 1e-8

    r, t = torch.tensor(r_axis * r), torch.tensor(t_axis * t)  # N x 3, N x 3
    R = SO3_exp(r)
    T = torch.eye(4).unsqueeze(0).repeat(nsamples, 1, 1)
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    return T


def random_pose_pixel_points(
    p: float,
    xyz: torch.Tensor,
    pose: torch.Tensor,
    intr: torch.Tensor,
    size: Tuple[int],
    rng: Optional[np.random.Generator] = None,
    tau: Optional[np.ndarray] = None
):
    cam_target = ready_camera(intr, size, pose)
    fx, fy, H, W = (
        cam_target.intrinsics.fx,
        cam_target.intrinsics.fy,
        cam_target.intrinsics.H,
        cam_target.intrinsics.W,
    )
    fx, fy = fx.detach().cpu().numpy(), fy.detach().cpu().numpy()
    xyz = cam_target.world_to_image_filtered(xyz)
    x, y, z = xyz[:, 0] - W / 2 + 0.5, xyz[:, 1] - H / 2 + 0.5, xyz[:, 2]
    x, y, Z0 = (
        x.detach().cpu().numpy(),
        y.detach().cpu().numpy(),
        z.detach().cpu().numpy(),
    )

    coefs = square_err(x, y, Z0, fx, fy, np.ones(3), np.ones(3))
    S = big_cov_err(coefs)

    val, vec = np.linalg.eigh(S)

    if tau is None:
        tau = (
            np.random.random(val.shape[0]) - 0.5
            if rng is None
            else rng.random(val.shape[0]) - 0.5
        )
    tau = tau / np.linalg.norm(tau)
    tau = p * tau / np.sqrt(val)
    tau = vec @ tau

    r, t = torch.tensor(tau[:3]), torch.tensor(tau[3:])  # N x 3, N x 3
    R = SO3_exp(r.unsqueeze(0)).squeeze(0)
    T = torch.eye(4, dtype=cam_target.extrinsics.view_matrix.dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def compute_scale(
    xyz: torch.Tensor,
    pose: torch.Tensor,
    intr: torch.Tensor,
    size: Tuple[int],
    mode: Literal["hs", "max", "mean"] = "hs",
) -> torch.Tensor:
    cam_target = ready_camera(intr, size, pose)
    fx, fy, H, W = (
        cam_target.intrinsics.fx,
        cam_target.intrinsics.fy,
        cam_target.intrinsics.H,
        cam_target.intrinsics.W,
    )
    fx, fy = fx.detach().cpu().numpy(), fy.detach().cpu().numpy()

    xyz = cam_target.world_to_image_filtered(xyz)
    x, y, z = xyz[:, 0] - W / 2 + 0.5, xyz[:, 1] - H / 2 + 0.5, xyz[:, 2]
    x, y, Z0 = (
        x.detach().cpu().numpy(),
        y.detach().cpu().numpy(),
        z.detach().cpu().numpy(),
    )

    coefs = square_err(x, y, Z0, fx, fy, np.ones(3), np.ones(3))
    S = big_cov_err(coefs)

    Sr = S[:3, :3]
    St = S[3:, 3:]

    valsr, _ = np.linalg.eigh(Sr)
    valst, _ = np.linalg.eigh(St)

    if mode == "max":
        scale = np.sqrt(np.max(np.abs(valsr)) / np.max(np.abs(valst)))
    elif mode == "mean":
        scale = np.sqrt(valsr.mean() / valst.mean())
    elif mode == "hs":
        scale = np.power((valsr**2).sum() / (valst**2).sum(), 1 / 4)
    else:
        raise ValueError(f"Invalide mode {mode}")
    scale = torch.tensor(scale, dtype=pose.dtype, device=pose.device)
    return scale
