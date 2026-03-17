import torch


def batched_conv(x, wb, conv_shape, ps_layer):
    B = wb.size(0)
    ch_out, ch_in, cur_ks, pad = conv_shape
    conv_weights = wb[:,:-1].view((B, ch_in, cur_ks, cur_ks, ch_out))
    conv_weights = conv_weights.permute(0,-1,1,2,3).flatten(end_dim=1) # (B, ch_out, ch_in, cur_ks, cur_ks) -> (B*ch_out, ch_in, cur_ks, cur_ks)
    conv_bias = wb[:,-1:].flatten() # B*ch_out
    x = torch.nn.functional.conv2d(x, conv_weights, conv_bias, stride=1, padding=pad, groups=B)
    return ps_layer(x)