import torch
import torch.nn as nn
import torch.nn.functional as F

from refiner_C.ops import hist_conv, sample_conv
    
class HOGLayer(nn.Module):
    def __init__(
        self,
        nbins: int,
        cell_size: int,
        cells_per_block: int,
        block_cells_stride: int = 3,
        transform_sqrt: bool = False,
        gradient_sign: bool = False,
        device="cpu",
        dtype=torch.float32,
    ):
        super(HOGLayer, self).__init__()

        # Preprocessing
        self.transform_sqrt = transform_sqrt

        # Gradients
        self.device = device
        self.dtype = dtype

        # fmt: off
        mat = 1 / 4 * torch.tensor(
                [
                    [1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]
                ],
                device=self.device,
                dtype=self.dtype)
        # fmt: on

        mat = torch.cat((mat[None], mat.t()[None]), dim=0)  # 2 x 3 x 3
        self.weight = mat.unsqueeze(1)
        self.gradient_sign = gradient_sign

        # Sampling initialization
        self.cell_size = cell_size
        self.cells_per_block = cells_per_block
        self.block_cells_stride = block_cells_stride
        
        # Histogram
        min_angle = -torch.pi if self.gradient_sign else 0
        bins_centers = torch.linspace(
            min_angle, torch.pi, nbins + 1, dtype=self.dtype, device=self.device
        )
        bins_centers = bins_centers[:-1]  # B
        m = (bins_centers[1] - bins_centers[0]) / 2
        bins_centers = bins_centers + m  # B
        self.bins_centers = bins_centers
        self.bandwidth = ((self.bins_centers[1] - self.bins_centers[0]) / 1.5).detach().cpu().item()
        self.eps = 1e-5

    def forward_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """
        mask: H x W
        """
        weight = torch.ones((1, 1, self.cell_size, self.cell_size), dtype = mask.dtype, device = mask.device) / (self.cell_size * self.cell_size)
        mask = F.conv2d(mask.unsqueeze(0).unsqueeze(0), weight, stride = self.cell_size, padding = 0)[0] # N x Hc x Wc
        mask = sample_conv(mask, self.cells_per_block, self.block_cells_stride)
        return torch.moveaxis(mask, 0, -1) # Hb x Wb x cpby x cpbx x 1
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Expanding dimensions
        ndimsx = len(x.shape)
        if ndimsx > 3 or ndimsx < 2:
            raise ValueError(f"input has {ndimsx} dimension, should be between 2 and 3")
        elif ndimsx == 2:
            x = x.unsqueeze(0)  # 1 x H x W
        if ndimsx == 3:
            x = x.unsqueeze(1)  # C x 1 x H x W

        # Gamma normalization
        if self.transform_sqrt:
            x = torch.sqrt(x)
        x = F.conv2d(x, self.weight, padding=1)  # C x 2 x H x W
        x[:, 0, :, 0], x[:, 0, :, -1], x[:, 1, 0, :], x[:, 1, -1, :] = 0, 0, 0, 0

        # Gradient computation
        mag = x.norm(dim=1)  # N x H x W
        phase = torch.atan2(x[:, 1, :, :], x[:, 0, :, :] + self.eps)  # N x H x W
        if not self.gradient_sign:
            phase = phase % torch.pi
        
        # Cell histograms
        cells_hists = hist_conv(phase, mag, self.bins_centers, self.cell_size, self.bandwidth) # N x Hc x Wc x B
        cells_hists = cells_hists.moveaxis(3, 1).contiguous()  # N x B x Hc x Wc if using compile, contigous is required for consistent stride

        # Block normalization
        N, B, Hc, Wc = cells_hists.shape
        cells_hists = cells_hists.reshape(-1, Hc, Wc) # N*B x Hc x Wc
        cells_hists = sample_conv(cells_hists, self.cells_per_block, self.block_cells_stride) # N*B x Hb x Wb x cpby x cpbx
        cells_hists = cells_hists.reshape(N, B, *cells_hists.shape[1:]) # N x B x Hb x Wb x cpby x cpbx
        cells_hists = cells_hists.moveaxis(1, -1)  # N x Hb x Wb x cpby x cpbx x B

        norms = torch.sqrt((cells_hists.flatten(3, 5) ** 2).mean())  # N x Hb x Wb x 1
        norms = norms.unsqueeze(-1).unsqueeze(-1)  # N x Hb x Wb x 1 x 1 x 1
        cells_hists = cells_hists / norms  # N x Hb x Wb x cpby x cpbx x B

        return cells_hists  # N x Hb x Wb x cpby x cpbx x B

