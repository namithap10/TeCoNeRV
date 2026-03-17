import math
import os

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils.quantize import *


def init_wb(out_ch, in_ch, ks):
    weight = torch.empty(in_ch, out_ch, ks, ks)
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

    bias = torch.empty(out_ch)
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    nn.init.uniform_(bias, -bound, bound)

    wb_list = [weight.permute(0, 2, 3, 1).flatten(end_dim=-2), bias[None]]
    return torch.cat(wb_list, dim=0).detach()


@register('nerv_enc_full_res')
class HyperNeRV(nn.Module):

    def __init__(self, tokenizer, hyponet, n_tokens, token_dims, transformer_encoder):
        super().__init__()
        dim = transformer_encoder['args']['dim']
        self.embed_size = 0
        self.tokenizer = models.make(tokenizer, args={'dim': dim})
        self.hyponet = models.make(hyponet, args={'n_groups': None})
        print(
            f'nerv_enc_full_res: hyponet param shapes: {self.hyponet.param_shapes}')
        self.transformer_encoder = models.make(transformer_encoder)

        self.base_params = nn.ParameterDict()
        n_wtokens = 0
        self.wtoken_postfc = nn.ModuleDict()
        self.wtoken_rng = dict()

        n_tokens = [int(x) for x in n_tokens.split('_')]
        token_dims = [int(x) for x in token_dims.split('_')]

        unique_params = 0
        i = 0
        for name, shape in self.hyponet.param_shapes.items():
            out_ch, in_ch, ks = shape
            self.base_params[name] = nn.Parameter(init_wb(out_ch, in_ch, ks))
            g = n_tokens[i]
            if g > 0:
                assert out_ch % g == 0
                self.wtoken_postfc[name] = nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, token_dims[i]),
                )
            self.wtoken_rng[name] = (n_wtokens, n_wtokens + g)
            n_wtokens += g
            unique_params += g * token_dims[i]
            i += 1
        self.wtokens = nn.Parameter(torch.randn(n_wtokens, dim))

        print(f'num unique params: {unique_params}')

    @classmethod
    def from_checkpoint(cls, ckpt, load_state_dict=True):
        if isinstance(ckpt, str):
            assert os.path.exists(ckpt), f'checkpoint {ckpt} does not exist'
            ckpt = torch.load(ckpt, map_location=lambda storage, loc: storage)
        else:
            assert isinstance(
                ckpt, dict), f'checkpoint must be a dict or a path to a checkpoint'

        tokenizer = ckpt['model']['args']['tokenizer']
        hyponet = ckpt['model']['args']['hyponet']
        n_tokens = ckpt['model']['args']['n_tokens']
        token_dims = ckpt['model']['args']['token_dims']
        transformer_encoder = ckpt['model']['args']['transformer_encoder']
        model = cls(tokenizer, hyponet, n_tokens,
                    token_dims, transformer_encoder)
        if load_state_dict:
            model.load_state_dict(ckpt['model']['sd'])
        return model

    def get_x(self, data):
        dtokens = self.tokenizer(data)
        B = dtokens.shape[0]
        wtokens = einops.repeat(self.wtokens, 'n d -> b n d', b=B)

        params = dict()
        trans_out = self.transformer_encoder(
            torch.cat([dtokens, wtokens], dim=1))
        weight_out = trans_out[:, -len(self.wtokens):]

        for name, _ in self.hyponet.param_shapes.items():
            l, r = self.wtoken_rng[name]
            x = self.wtoken_postfc[name](weight_out[:, l: r, :])
            params[name] = x

        return weight_out, params

    def forward(self, data, x_only=False, quant_bit=32):
        if x_only:
            return self.get_x(data)

        dtokens = self.tokenizer(data)
        B = dtokens.shape[0]
        wtokens = einops.repeat(self.wtokens, 'n d -> b n d', b=B)

        params = dict()
        trans_out = self.transformer_encoder(
            torch.cat([dtokens, wtokens], dim=1))
        weight_out = trans_out[:, -len(self.wtokens):]
        params['embed'] = None
        pre_mod = dict()

        hyponet_bits, quant_overhead_bits = 0, 0
        for name, shape in self.hyponet.param_shapes.items():
            wb = einops.repeat(self.base_params[name], 'n m -> b n m', b=B)
            init_w, init_b = wb[:, :-1, :], wb[:, -1:, :]

            l, r = self.wtoken_rng[name]
            if r - l > 0:
                x = self.wtoken_postfc[name](weight_out[:, l: r, :])
                x = x.transpose(-1, -2)  # (B, in_ch, g)

                if quant_bit < 32:  # Quantization aware training, disabled by default
                    x_quant_v, x_scale, x_t_min = quantize_per_tensor(
                        x, bit=quant_bit, axis=1, dither=False)
                    x = x_t_min.to(torch.float32) + \
                        (x_scale.to(torch.float32) * x_quant_v)
                    quant_overhead_bits += ((x_scale.numel() +
                                            x_t_min.numel()) * 32 / x.shape[0])

                x = x.transpose(-1, -2)  # (B, g, in_ch)
                pre_mod[name] = x.clone()
                repeat_num = init_w.nelement() // x.nelement()
                # (B, in_ch, out_ch)
                x = einops.repeat(x, 'B n m -> B n d m',
                                  d=repeat_num).reshape_as(init_w)
                hyponet_bits += ((x.numel() * quant_bit) / x.shape[0])

                w = F.normalize(init_w * x, dim=1)
            else:
                pre_mod[name] = None
                w = F.normalize(init_w, dim=1)

            wb = torch.cat([w, init_b], dim=1)
            params[name] = wb

        self.hyponet.set_params(params)
        return {
            'hyponet': self.hyponet,
            'hyponet_bits': hyponet_bits,
            'quant_overhead_bits': quant_overhead_bits,
            'x_dict': pre_mod
        }
