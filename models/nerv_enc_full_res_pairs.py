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


@register('nerv_enc_full_res_pairs')
class NeRVEncFullResPairs(nn.Module):

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

    def process_single_clip(self, x, init_w, quant_bit, B):
        """Helper function to process a single clip's parameters."""
        quant_overhead_bits = 0
        x = x.transpose(-1, -2)  # (B, token_dim, g)

        if quant_bit < 32:  # Quantization aware training, off by default
            x_quant_v, x_scale, x_t_min = quantize_per_tensor(
                x, bit=quant_bit, axis=1, dither=False)
            x = x_t_min.to(torch.float32) + \
                (x_scale.to(torch.float32) * x_quant_v)
            # Divide by B for per-sample average
            quant_overhead_bits = (
                (x_scale.numel() + x_t_min.numel()) * 32 / x.shape[0])

        x_pre_mod = x.transpose(-1, -2).clone()  # (B, g, token_dim)

        repeat_num = init_w.nelement() // x_pre_mod.nelement()
        x_reshaped = einops.repeat(
            x_pre_mod, 'B n m -> B n d m', d=repeat_num).reshape_as(init_w)

        return x_reshaped, x_pre_mod, quant_overhead_bits

    def forward(self, data, x_only=False, quant_bit=32):
        if x_only:
            raise NotImplementedError(
                "x_only=True is not supported for pairs in this model yet.")

        inp = data['inp']

        assert inp.ndim == 6 and inp.shape[
            1] == 2, "Input should be pairs (B, 2, C, T, H, W)"

        data1 = {'inp': inp[:, 0]}
        data2 = {'inp': inp[:, 1]}

        dtokens1 = self.tokenizer(data1)
        dtokens2 = self.tokenizer(data2)
        B = dtokens1.shape[0]  # Batch size

        wtokens1 = einops.repeat(self.wtokens, 'n d -> b n d', b=B)
        wtokens2 = einops.repeat(self.wtokens, 'n d -> b n d', b=B)

        params1, params2 = dict(), dict()
        pre_mod1, pre_mod2 = dict(), dict()

        # Process first clip of the pair
        trans_out1 = self.transformer_encoder(
            torch.cat([dtokens1, wtokens1], dim=1))
        weight_out1 = trans_out1[:, -len(self.wtokens):]
        # Process second clip of the pair
        trans_out2 = self.transformer_encoder(
            torch.cat([dtokens2, wtokens2], dim=1))
        weight_out2 = trans_out2[:, -len(self.wtokens):]

        params1['embed'], params2['embed'] = None, None

        hyponet_bits, quant_overhead_bits = 0, 0
        for name, shape in self.hyponet.param_shapes.items():
            wb = einops.repeat(self.base_params[name], 'n m -> b n m', b=B)
            init_w, init_b = wb[:, :-1, :], wb[:, -1:, :]

            l, r = self.wtoken_rng[name]
            if r - l > 0:
                x1_raw = self.wtoken_postfc[name](
                    weight_out1[:, l: r, :])  # Shape: (B, g, token_dim)
                x2_raw = self.wtoken_postfc[name](
                    weight_out2[:, l: r, :])  # Shape: (B, g, token_dim)

                unique_x1, pre_mod1[name], quant_overhead_bits1 = self.process_single_clip(
                    x1_raw, init_w, quant_bit, B)
                unique_x2, pre_mod2[name], quant_overhead_bits2 = self.process_single_clip(
                    x2_raw, init_w, quant_bit, B)

                hyponet_bits += ((unique_x1.numel() *
                                 quant_bit) / unique_x1.shape[0])
                quant_overhead_bits += quant_overhead_bits1

                w1 = F.normalize(init_w * unique_x1, dim=1)
                w2 = F.normalize(init_w * unique_x2, dim=1)
            else:  # No learnable weight tokens for this layer
                pre_mod1[name] = None
                pre_mod2[name] = None
                w1 = F.normalize(init_w, dim=1)
                w2 = F.normalize(init_w, dim=1)

            wb1 = torch.cat([w1, init_b], dim=1)
            wb2 = torch.cat([w2, init_b], dim=1)
            params1[name] = wb1
            params2[name] = wb2

        # Set the model's hyponetwork using the parameters for the first clip
        self.hyponet.set_params(params1)
        return {
            'hyponet_bits': hyponet_bits,
            'quant_overhead_bits': quant_overhead_bits,
            'hyponet': self.hyponet,
            'params1': params1,
            'params2': params2,
            'x_dict1': pre_mod1,
            'x_dict2': pre_mod2
        }
