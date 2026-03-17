import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register


@register('vidrec_tokenizer')
class VidrecTokenizer(nn.Module):

    def __init__(self, input_size, patch_size, dim, frame_num=16,
            eval_frames='none', padding=0, img_channels=3, img_groups=1):
        super().__init__()
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        if isinstance(padding, int):
            padding = (padding, padding)
        input_size = input_size
        self.patch_size = patch_size
        self.padding = padding
        self.prefc = nn.Linear(patch_size[0] * patch_size[1] * img_channels * img_groups, dim)
        n_patches = 1
        for dim_id in range(2):
            n_patches *= (input_size[dim_id] + self.padding[dim_id] * 2) // self.patch_size[dim_id]
        clip_num = frame_num // img_groups
        self.posemb = nn.Parameter(torch.randn(clip_num, n_patches, dim))
        self.img_groups = img_groups

    def forward(self, data):
        if isinstance(data, dict):
            x = data['inp']
        else:
            x = data
        p = self.patch_size
        x_batch, c, t, h, w = x.size()
        x = x.view(x_batch, c, -1, self.img_groups, h, w).permute(0,2,1,3,-2,-1)
        x = F.unfold(x.flatten(end_dim=1).flatten(start_dim=1,end_dim=2), p, stride=p, padding=self.padding) # (B*T, C * img_groups * p * p, L)
        x = x.view((x_batch, -1)+x.shape[1:]).permute(0, 1, 3, 2).flatten(start_dim=1, end_dim=2).contiguous()
        x = self.prefc(x) + self.posemb.flatten(end_dim=1).unsqueeze(0)
        return x