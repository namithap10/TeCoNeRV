import torch.nn as nn


class PixelShuffleRect(nn.Module):
    def __init__(self, upscale_h, upscale_w):
        super(PixelShuffleRect, self).__init__()
        self.upscale_h = upscale_h
        self.upscale_w = upscale_w

    def __repr__(self):
        return f"{self.__class__.__name__}(upscale_h={self.upscale_h}, upscale_w={self.upscale_w})"

    def forward(self, x):
        batch_sz, channels, in_h, in_w = x.size()
        out_channels = channels // (self.upscale_h * self.upscale_w)
        out_height, out_width = in_h * self.upscale_h, in_w * self.upscale_w

        x = x.view(batch_sz, out_channels, self.upscale_h, self.upscale_w, in_h, in_w)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(batch_sz, out_channels, out_height, out_width)
        return x