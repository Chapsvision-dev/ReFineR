import torch
from torch import Tensor

from refiner.utils.lie import skew_sym


class So3_left_action(torch.autograd.Function):
    @staticmethod
    def forward(R: Tensor, points: Tensor, r: Tensor):
        """
        Lie group action of s03 on R3 given by r.P = Exp(r)P
        Inputs:
            r: Batched tensor or rotations (C x 3) defined by the 3 rotations (rx, ry, rz
            points: Batched tensor of points to transform (N x 3)
        Output:
            Batched array of transformed points (N x C x 3)
        """

        points = torch.matmul(R, points.T)  # (C x 3 x 3 @ 3 x N) = C x 3 x N
        points = torch.moveaxis(points, 2, 0)  # N x C x 3

        return points  # N x C x 3

    @staticmethod
    def setup_context(ctx, inputs, output):
        R, _, _ = inputs
        points = output
        ctx.save_for_backward(R, points)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        """
        Derivative of the action is the classic derivative w.r.t P and the Lie derivative to the left w.r.t r
        d/dp = Exp(r) = R
        d/dr = -(Exp(r)P)x
        input:
            grad_output: gradients of the output tensor (N x C x 3)
        output:
            grad_r: gradients w.r.t tau (C x 6)
            grad_points: gradients w.r.t points (N x 3)
        """
        R, output = ctx.saved_tensors
        grad_r, grad_points = None, None

        if ctx.needs_input_grad[2]:
            C, N = output.shape[1], output.shape[0]
            output = -skew_sym(output.reshape(-1, 3))
            output = output.reshape(N, C, 3, 3)  # N x C x 3 x 3
            grad_r = grad_output.unsqueeze(-1)  # N x C x 3 x 1
            grad_r = grad_r * output  # N x C x 3 x 3
            grad_r = grad_r.sum(dim=(0, 2))  # C x 3

        if ctx.needs_input_grad[1]:
            grad_points = grad_output.unsqueeze(-1) * R.unsqueeze(0)  # N x C x 3 x 3
            grad_points = grad_points.sum(dim=(1, 2))  # N x 3

        return None, grad_points, grad_r


class Se3_left_action(torch.autograd.Function):
    @staticmethod
    def forward(T: Tensor, points: Tensor, tau: Tensor):
        """
        Lie group action of se3 on R3 given by tau.P = Exp(r)P + V(r)rho
        Inputs:
            tau: Batched tensor or transforms (C x 6) defined by the 3 rotations and 3 translations (rhox, rhoy, rhoz, rx, ry, rz)
            points: Batched tensor of points to transform (N x 3)
        Output:
            Batched array of transformed points (N x C x 3)
        """

        points = torch.concatenate(
            (
                points,
                torch.ones(
                    (points.shape[0], 1), dtype=points.dtype, device=points.device
                ),
            ),
            axis=1,
        )  # N x 4
        points = torch.matmul(T, points.T)  # (C x 4 x 4 @ 4 x N) = C x 4 x N
        points = torch.moveaxis(points, 2, 0)  # N x C x 4

        return points[:, :, :3]  # N x C x 3

    @staticmethod
    def setup_context(
        ctx: torch.autograd.Function, inputs: torch.Tensor, output: torch.Tensor
    ):
        T, _, _ = inputs
        points = output
        ctx.save_for_backward(T, points)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        """
        Derivative of the action is the classic derivative w.r.t P and the Lie derivative to the left w.r.t tau
        d/dp = R
        d/dtau = [-(RP + t)x I3]
        input:
            grad_output: gradients of the output tensor (N x C x 3)
        output:
            grad_tau: gradients w.r.t tau (C x 6)
            grad_points: gradients w.r.t points (N x 3)
        """
        T, output = ctx.saved_tensors
        grad_tau, grad_points = None, None

        if ctx.needs_input_grad[2]:
            N, C = output.shape[0], output.shape[1]
            output = -skew_sym(output.reshape(-1, 3)).reshape(
                N, C, 3, 3
            )  # N x C x 3 x 3
            id = (
                torch.eye(3, dtype=output.dtype, device=output.device)
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(N, C, 1, 1)
            )  # N x C x 3 x 3
            output = torch.concatenate((id, output), axis=3)  # N x C x 3 x 6
            grad_tau = grad_output.unsqueeze(-1)  # N x C x 3 x 1
            grad_tau = grad_tau * output  # N x C x 3 x 6
            grad_tau = grad_tau.sum(dim=(0, 2))  # C x 6

        if ctx.needs_input_grad[1]:
            R = T[:, :3, :3].unsqueeze(0)  # 1 x C x 3 x 3
            grad_points = grad_output.unsqueeze(-1) * R  # N x C x 3 x 3
            grad_points = grad_points.sum(dim=(1, 2))  # N x 3

        return None, grad_points, grad_tau

class So3_right_action(torch.autograd.Function):
    @staticmethod
    def forward(R: Tensor, points: Tensor, r: Tensor):
        """
        Lie group action of s03 on R3 given by r.P = Exp(r)P
        Inputs:
            r: Batched tensor or rotations (C x 3) defined by the 3 rotations (rx, ry, rz
            points: Batched tensor of points to transform (N x 3)
        Output:
            Batched array of transformed points (N x C x 3)
        """

        points = torch.matmul(R, points.T)  # (C x 3 x 3 @ 3 x N) = C x 3 x N
        points = torch.moveaxis(points, 2, 0)  # N x C x 3

        return points  # N x C x 3

    @staticmethod
    def setup_context(ctx, inputs, output):
        R, points, _ = inputs
        ctx.save_for_backward(R, points)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        """
        Derivative of the action is the classic derivative w.r.t P and the Lie derivative to the left w.r.t r
        d/dp = Exp(r) = R
        d/dr = -Exp(r)Px
        input:
            grad_output: gradients of the output tensor (N x C x 3)
        output:
            grad_r: gradients w.r.t tau (C x 6)
            grad_points: gradients w.r.t points (N x 3)
        """
        R, input = ctx.saved_tensors
        grad_r, grad_points = None, None

        if ctx.needs_input_grad[2]:
            input = skew_sym(input) # N x 3 x 3
            input = -torch.matmul(R[None, :, :, :], input[:, None, :, :]) # 1 x C x 3 x 3 @ N x 1 x 3 x 3 = N x C x 3 x 3
            grad_r = grad_output.unsqueeze(-1)  # N x C x 3 x 1
            grad_r = grad_r * input  # N x C x 3 x 3
            grad_r = grad_r.sum(dim=(0, 2))  # C x 3

        if ctx.needs_input_grad[1]:
            grad_points = grad_output.unsqueeze(-1) * R.unsqueeze(0)  # N x C x 3 x 3
            grad_points = grad_points.sum(dim=(1, 2))  # N x 3

        return None, grad_points, grad_r
class Se3_right_action(torch.autograd.Function):
    @staticmethod
    def forward(T: Tensor, points: Tensor, tau: Tensor):
        """
        Lie group action of se3 on R3 given by tau.P = Exp(r)P + V(r)rho
        Inputs:
            tau: Batched tensor or transforms (C x 6) defined by the 3 rotations and 3 translations (rhox, rhoy, rhoz, rx, ry, rz)
            points: Batched tensor of points to transform (N x 3)
        Output:
            Batched array of transformed points (N x C x 3)
        """

        points = torch.concatenate(
            (
                points,
                torch.ones(
                    (points.shape[0], 1), dtype=points.dtype, device=points.device
                ),
            ),
            axis=1,
        )  # N x 4
        points = torch.matmul(T, points.T)  # (C x 4 x 4 @ 4 x N) = C x 4 x N
        points = torch.moveaxis(points, 2, 0)  # N x C x 4

        return points[:, :, :3]  # N x C x 3

    @staticmethod
    def setup_context(
        ctx: torch.autograd.Function, inputs: torch.Tensor, output: torch.Tensor
    ):
        T, points, _ = inputs
        ctx.save_for_backward(T, points)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        """
        Derivative of the action is the classic derivative w.r.t P and the Lie derivative to the right w.r.t tau
        d/dp = R
        d/dtau = [R, -RPx]
        input:
            grad_output: gradients of the output tensor (N x C x 3)
        output:
            grad_tau: gradients w.r.t tau (C x 6)
            grad_points: gradients w.r.t points (N x 3)
        """
        T, input = ctx.saved_tensors
        grad_tau, grad_points = None, None

        if ctx.needs_input_grad[2]:
            N = input.shape[0]
            input = skew_sym(input) # N x 3 x 3
            input = -torch.matmul(T[None, :, :3, :3], input[:, None, :, :]) # 1 x C x 3 x 3 @ N x 1 x 3 x 3 = N x C x 3 x 3
            R = T[None, :, :3, :3].repeat(N, 1, 1, 1)  # N x C x 3 x 3
            output = torch.concatenate((R, input), axis=3)  # N x C x 3 x 6
            grad_tau = grad_output.unsqueeze(-1)  # N x C x 3 x 1
            grad_tau = grad_tau * output  # N x C x 3 x 6
            grad_tau = grad_tau.sum(dim=(0, 2))  # C x 6

        if ctx.needs_input_grad[1]:
            R = T[:, :3, :3].unsqueeze(0)  # 1 x C x 3 x 3
            grad_points = grad_output.unsqueeze(-1) * R  # N x C x 3 x 3
            grad_points = grad_points.sum(dim=(1, 2))  # N x 3

        return None, grad_points, grad_tau
