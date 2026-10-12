import torch


def up_to_homogeneous(x: torch.Tensor) -> torch.Tensor:
    """Up-projects to homogeneous coordinates.
    x: N x D
    return: N x (D+1) where the last item of the last dimension is a 1
    """
    return torch.cat([x, torch.ones_like(x[..., 0:1])], dim=-1)


def down_from_homogeneous(y: torch.Tensor) -> torch.Tensor:
    """Perspective projection
    y: N x (D+1)
    return: N x D
    """
    return y[..., :-1] / y[..., -1:]


def min_max_scale(t, channels=None):
    """
    min max scaler among all dimensions except the last one
    t: ..., D
    """
    s = t.shape
    D = s[-1]
    channels = torch.arange(D) if channels is None else channels

    reduced_shape = [1 for _ in range(len(s) - 1)] + [D]
    m = torch.min(t.reshape(-1, D), dim=0)[0].reshape(reduced_shape)  # 1 x ... x 1 x D
    M = torch.max(t.reshape(-1, D), dim=0)[0].reshape(reduced_shape)  # 1 x ... x 1 x D
    for c in channels:
        t[..., c] = (t[..., c] - m[..., c]) / (M[..., c] - m[..., c])

    return t
