import torch

def inverse_transform(T: torch.Tensor) -> torch.Tensor:
    """
    T: N x 3 x 3
    """
    R = T[:, :3, :3] # N x 3 x 3
    t = T[:, :3, [3]] # N x 3 x 1
    inv_T = torch.zeros_like(T)
    inv_T[:, 3, 3] = 1.
    Rt = torch.moveaxis(R, 2, 1)
    inv_T[:, :3, :3] = Rt
    inv_T[:, :3, [3]] = -torch.matmul(Rt, t) # N x 3 x 1
    return inv_T

def skew_sym(x: torch.Tensor) -> torch.Tensor:
    """
    Converts to skew matrix
    x: N x 3
    output: N x 3 x 3
    """
    device = x.device
    dtype = x.dtype
    N = x.shape[0]
    ssm = torch.zeros(N, 3, 3, device=device, dtype=dtype)
    # fmt: off
    ssm[:, 0, 1] = -x[:, 2]
    ssm[:, 0, 2] =  x[:, 1]
    ssm[:, 1, 0] =  x[:, 2]
    ssm[:, 1, 2] = -x[:, 0]
    ssm[:, 2, 0] = -x[:, 1]
    ssm[:, 2, 1] =  x[:, 0]
    # fmt: on
    return ssm


def inv_skew_sym(ssm: torch.Tensor) -> torch.Tensor:
    """
    Converts a skew matrix to its vector form [w]_x
    ssm: N x 3 x 3
    output: N x 3
    """
    return ssm[:, [2, 0, 1], [1, 2, 0]]


def V(theta: torch.Tensor) -> torch.Tensor:
    """
    theta: N x 3
    output: N x 3 x 3
    """
    eps = 1e-5
    dtype = theta.dtype
    device = theta.device

    N = theta.shape[0]
    W = skew_sym(theta)  # N x 3 x 3
    W2 = W @ W  # N x 3 x 3
    angle = torch.norm(theta, dim=1)  # N
    I = (
        torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(N, 1, 1)
    )  # N x 3 x 3

    fil = torch.abs(angle) > eps
    fil_angle = angle[fil].unsqueeze(-1).unsqueeze(-1).repeat((1, 3, 3))
    tayl_angle = angle[~fil].unsqueeze(-1).unsqueeze(-1).repeat((1, 3, 3))

    I[fil] = (
        I[fil]
        + ((1.0 - torch.cos(fil_angle)) / (fil_angle**2)) * W[fil]
        + ((fil_angle - torch.sin(fil_angle)) / (fil_angle**3)) * W2[fil]
    )
    I[~fil] = (
        I[~fil]
        + (1 / 2 - (tayl_angle**2) / 24) * W[~fil]
        + (1 / 6 - (tayl_angle**2) / 120) * W2[~fil]
    )

    return I

def inv_V(theta: torch.Tensor) -> torch.Tensor:
    """
    theta: N x 3
    output: N x 3 x 3
    """
    eps = 1e-5
    dtype = theta.dtype
    device = theta.device

    N = theta.shape[0]
    W = skew_sym(theta)  # N x 3 x 3
    W2 = W @ W  # N x 3 x 3
    angle = torch.norm(theta, dim=1)  # N
    I = (
        torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(N, 1, 1)
    )  # N x 3 x 3

    fil = torch.abs(angle) > eps
    fil_angle = angle[fil].unsqueeze(-1).unsqueeze(-1).repeat((1, 3, 3))
    tayl_angle = angle[~fil].unsqueeze(-1).unsqueeze(-1).repeat((1, 3, 3))

    I[fil] = (
        I[fil]
        - 0.5 * W[fil]
        + (1 / fil_angle ** 2 - (1 + torch.cos(fil_angle)) / (2 * fil_angle * torch.sin(fil_angle))) * W2[fil]
    )
    I[~fil] = (
        I[~fil]
        - 0.5 * W[~fil]
        + (1 / 12 - 29 * (tayl_angle**2) / 720) * W2[~fil]
    )

def SO3_exp(theta: torch.Tensor) -> torch.Tensor:
    """:
    Computes SO3 exponential from axis angles theta
    theta: N x 3
    output: N x 3 x 3
    """
    eps = 1e-5
    device = theta.device
    dtype = theta.dtype

    N = theta.shape[0]
    W = skew_sym(theta)  # N x 3 x 3
    W2 = W @ W  # N x 3 x 3
    angle = torch.linalg.norm(theta, dim=1)  # N
    I = (
        torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(N, 1, 1)
    )  # N x 3 x 3

    fil = torch.abs(angle) > eps
    fil_angle = angle[fil].unsqueeze(-1).unsqueeze(-1).repeat((1, 3, 3))
    tayl_angle = angle[~fil].unsqueeze(-1).unsqueeze(-1).repeat((1, 3, 3))

    I[fil] = (
        I[fil]
        + (torch.sin(fil_angle) / fil_angle) * W[fil]
        + ((1 - torch.cos(fil_angle)) / (fil_angle**2)) * W2[fil]
    )
    I[~fil] = (
        I[~fil]
        + (1 - (tayl_angle**2) / 6) * W[~fil]
        + (1 / 2 - (tayl_angle**2) / 24) * W2[~fil]
    )

    return I


def SO3_log(R: torch.Tensor) -> torch.Tensor:
    """
    SO3 log from batch of rotations
    R: N x 3 x 3
    output: N x 3
    """
    eps = 1e-5

    theta = torch.arccos((torch.einsum("...ii", R) - 1) / 2).unsqueeze(
        -1
    )  # N x 1 (in [0, pi])
    i = inv_skew_sym(R - torch.transpose(R, 1, 2))
    if theta > eps:
        theta = theta / (2 * torch.sin(theta)) * i  # N x 3
    else:
        theta = (1 / 2 + (theta**2) / 12) * i
    return theta


def SE3_exp(tau: torch.Tensor) -> torch.Tensor:
    """
    SE3 exponential from batch parameters tau
    tau: N x 6
    output: N x 4 x 4
    """
    dtype = tau.dtype
    device = tau.device

    N = tau.shape[0]
    rho = tau[:, :3]  # N x 3
    theta = tau[:, 3:]  # N x 3
    R = SO3_exp(theta)  # N x 3 x 3
    t = (V(theta) @ rho.unsqueeze(-1)).squeeze()  # N x 3 x 3 @ N X 3 x 1 = N x 3 x 1
    T = (
        torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(N, 1, 1)
    )  # N x 4 x 4
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    return T


def SE3_log(T: torch.Tensor) -> torch.Tensor:
    """
    SE3 log from batch of rigid transforms
    T: N x 4 x 4
    output: N x 6
    """
    R = T[:, :3, :3]  # N x 3 x 3
    t = T[:, :3, 3]  # N x 3
    theta = SO3_log(R)
    rho = torch.linalg.inv(V(theta)) @ t.unsqueeze(-1)
    rho = rho.squeeze(-1)  # N x 3 x 3 @ N x 3 x 1 = N x 3 x 1
    return torch.cat((rho, theta), dim=-1)  # N x 6
