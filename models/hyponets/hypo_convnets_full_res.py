import numpy as np
import torch
import torch.nn as nn

from models import register

from .layers import batched_conv
from .pixelshuffle_rect import PixelShuffleRect


@register('hypo_convnets_full_res')
class HypoConvnets(nn.Module):

    def __init__(self, in_dim, out_dim, hid_dim, strds_h, strds_w, n_groups, ks, use_pe, pe_dim,
        size='none', act='relu', out_bias=0, pe_sigma=1024):
        super().__init__()
        from math import ceil
        self.use_pe = use_pe
        self.pe_dim = pe_dim
        self.pe_sigma = pe_sigma

        if use_pe:
            last_dim = in_dim * pe_dim
        else:
            last_dim = in_dim
        strds_list_h = [int(x) for x in strds_h.split('_' if '_' in strds_h else ' ')]
        strds_list_w = [int(x) for x in strds_w.split('_' if '_' in strds_w else ' ')]
        depth = len(strds_list_h)
        ks_list = [int(x) for x in ks.split('_')]
        ks_list += ks_list[-1:] * (depth - len(ks_list))
        ch = hid_dim

        self.ps_layers = nn.ModuleList()
        self.param_shapes = dict()
        self.conv_shape_list = []
        for i, (cur_strd_h, cur_strd_w) in enumerate(zip(strds_list_h, strds_list_w)):
            cur_dim = ch if i < depth - 1 else out_dim
            cur_ks = ks_list[i]
            ch_out = cur_dim * cur_strd_h * cur_strd_w
            self.param_shapes[f'wb{i}'] = (ch_out, last_dim, cur_ks)
            cur_pad_h = 0 if (cur_ks == cur_strd_h and i == 0) else ceil(cur_ks - 1) // 2
            cur_pad_w = 0 if (cur_ks == cur_strd_w and i == 0) else ceil(cur_ks - 1) // 2
            self.conv_shape_list.append((ch_out, last_dim, cur_ks, (cur_pad_h, cur_pad_w))) 
            self.ps_layers.append(PixelShuffleRect(upscale_h=cur_strd_h, upscale_w=cur_strd_w))
            last_dim = cur_dim

        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'gelu':
            self.act = nn.GELU()
        else:
            NotImplementedError
        self.params = None
        self.out_bias = out_bias
        self.depth = depth

    def set_params(self, params):
        self.params = params

    def convert_posenc(self, x):
        w = torch.exp(torch.linspace(0, np.log(self.pe_sigma), self.pe_dim // 2, device=x.device)) # (PE_DIM/2,)
        x = torch.matmul(x.unsqueeze(-1), w.unsqueeze(0)).view(*x.shape[:-1], -1) # (B, T, PE_DIM/2)
        x = torch.cat([torch.cos(np.pi * x), torch.sin(np.pi * x)], dim=-1) # (B, T, PE_DIM)
        return x

    def OutImg(self, x):
        if self.out_bias == 'sigmoid':
            return torch.sigmoid(x)
        elif self.out_bias == 'tanh':
            return (torch.tanh(x) * 0.5) + 0.5
        else:
            return x + float(self.out_bias)

    def forward(self, x):
        B, T = x.size(0), x.size(1)
        
        if self.params['embed'] is not None:
            x = self.params['embed'].transpose(0,1).flatten(start_dim=1, end_dim=2)
        else:
            x = x.view(B, -1, x.shape[-1])
            if self.use_pe:
                x = self.convert_posenc(x)
            x = x.permute(1,0,2).reshape(T, -1, 1, 1)    # (T, B*PE_DIM, 1, 1)

        # import ipdb; ipdb.set_trace()
        for i in range(self.depth):
            x = batched_conv(x, self.params[f'wb{i}'], self.conv_shape_list[i], self.ps_layers[i])
            if i < self.depth - 1:
                x = self.act(x)
            else:
                x = self.OutImg(x)
        x = x.view((T, B, -1) + x.shape[-2:])
        return x

