from typing import Tuple
import torch

__all__ = ["hist_conv", "sample_conv"]

# Hist conv
def hist_conv(A: torch.Tensor, weight: torch.Tensor, bins: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    return torch.ops.refiner_C.hist_conv.default(A, weight, bins, kernel_size, sigma)

@torch.library.register_fake("refiner_C::hist_conv")
def _(A: torch.Tensor, weight: torch.Tensor, bins: torch.Tensor, kernel_size: int, sigma: float):
    C, H, W = A.shape[0], A.shape[1], A.shape[2]
    B = bins.shape[0]
    return torch.empty((C, H // kernel_size, W // kernel_size, B), dtype = A.dtype, device = A.device)

@torch.library.register_fake("refiner_C::hist_conv_backward")
def _(grad_A: torch.Tensor, A: torch.Tensor, weight: torch.Tensor, bins: torch.Tensor, kernel_size: int, sigma: float):
    return torch.empty_like(A), torch.empty_like(weight)

def hist_conv_setup_context(ctx, inputs, output):

    img, weights, bins, kernel_size, sigma = inputs
    kernel_size = torch.tensor(kernel_size, dtype = img.dtype, device = img.device)
    sigma = torch.tensor(sigma, dtype = img.dtype, device = img.device)
    ctx.save_for_backward(img, weights, bins, kernel_size, sigma)

def hist_conv_backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None, None]:
    """
    grad_output: C x H0 x W0 x B
    """
    img, weights, bins, kernel_size, sigma = ctx.saved_tensors
    kernel_size = kernel_size.detach().cpu().to(torch.int32).item()
    sigma = sigma.detach().cpu().item()
    grad_img, grad_weights = torch.ops.refiner_C.hist_conv_backward.default(grad_output, img, weights, bins, kernel_size, sigma)

    return grad_img, grad_weights, None, None, None

torch.library.register_autograd(
    "refiner_C::hist_conv", hist_conv_backward, setup_context=hist_conv_setup_context)

# Sample conv
def sample_conv(A: torch.Tensor, kernel_size: int, stride: int) -> torch.Tensor:
    return torch.ops.refiner_C.sample_conv.default(A, kernel_size, stride)

@torch.library.register_fake("refiner_C::sample_conv")
def _(A: torch.Tensor, kernel_size: int, stride: int):
    # C, (H - kernel_size) / stride + 1, (W - kernel_size) / stride + 1, kernel_size, kernel_siz
    C, H, W = A.shape[0], A.shape[1], A.shape[2]
    return torch.empty((C, (H - kernel_size) // stride + 1, (W - kernel_size) // stride + 1, kernel_size, kernel_size), dtype = A.dtype, device = A.device)

@torch.library.register_fake("refiner_C::sample_conv_backward")
def _(grad_A: torch.Tensor, A: torch.Tensor, kernel_size: int, stride: int):
    C, H, W = A.shape[0], A.shape[1], A.shape[2]
    return torch.empty((C, H, W))

def sample_conv_setup_context(ctx, inputs, output):

    img, kernel_size, stride = inputs
    kernel_size = torch.tensor(kernel_size, dtype = img.dtype, device = img.device)
    stride = torch.tensor(stride, dtype = img.dtype, device = img.device)
    ctx.save_for_backward(img, kernel_size, stride)

def sample_conv_backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None, None]:
    """
    grad_output: C x H0 x W0 x B
    """
    img, kernel_size, stride = ctx.saved_tensors
    kernel_size = kernel_size.detach().cpu().to(torch.int32).item()
    stride = stride.detach().cpu().to(torch.int32).item()
    grad_img = torch.ops.refiner_C.sample_conv_backward.default(grad_output, img, kernel_size, stride)
    return grad_img, None, None

torch.library.register_autograd(
    "refiner_C::sample_conv", sample_conv_backward, setup_context=sample_conv_setup_context)

# Target detection
def target_detect(A: torch.Tensor, radius: int, radiusin: int, min_color_std: int, max_offset: int, max_line_cov: float, min_angle: float, cross_thresh: float, thresh: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.refiner_C.target_detect.default(A, radius, radiusin, min_color_std, max_offset, max_line_cov, min_angle, cross_thresh, thresh)