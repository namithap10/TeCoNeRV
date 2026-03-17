
import torch


def quantize_per_tensor(t, bit=8, axis=-1, dither=False):
    if axis == -1:
        t_valid = t != 0
        if t_valid.sum() == 0:
            scale = torch.tensor(0).to(t.device)
            t_min = torch.tensor(0).to(t.device)
        else:
            t_min, t_max = t[t_valid].min(), t[t_valid].max()
            scale = (t_max - t_min) / 2**bit
    elif axis == 0:
        min_max_list = []
        for i in range(t.size(0)):
            t_valid = t[i] != 0
            if t_valid.sum():
                min_max_list.append([t[i][t_valid].min(), t[i][t_valid].max()])
            else:
                min_max_list.append([0, 0])
        min_max_tf = torch.tensor(min_max_list).to(t.device)
        scale = (min_max_tf[:, 1] - min_max_tf[:, 0]) / 2**bit
        if t.dim() == 4:
            scale = scale[:, None, None, None]
            t_min = min_max_tf[:, 0, None, None, None]
        elif t.dim() == 3:
            scale = scale[:, None, None]
            t_min = min_max_tf[:, 0, None, None]
        elif t.dim() == 2:
            scale = scale[:, None]
            t_min = min_max_tf[:, 0, None]
    elif axis == 1:
        min_max_list = []
        for i in range(t.size(1)):
            t_valid = t[:, i] != 0
            if t_valid.sum():
                min_max_list.append([t[:, i][t_valid].min(), t[:, i][t_valid].max()])
            else:
                min_max_list.append([0, 0])
        min_max_tf = torch.tensor(min_max_list).to(t.device)
        scale = (min_max_tf[:, 1] - min_max_tf[:, 0]) / 2**bit
        if t.dim() == 4:
            scale = scale[None, :, None, None]
            t_min = min_max_tf[None, :, 0, None, None]
        elif t.dim() == 3:
            scale = scale[None, :, None]
            t_min = min_max_tf[None, :, 0, None]
        elif t.dim() == 2:
            scale = scale[None, :]
            t_min = min_max_tf[None, :, 0]

    if dither:
        print("dithering before quant")
        # Calculate the noise range based on the scale
        noise_range = scale / 2
        # Generate uniform noise in the range [-noise_range, noise_range]
        noise = (torch.rand_like(t) * 2 - 1) * noise_range
        t = t + noise

    quant_t = ((t - t_min) / (scale + 1e-19)).round()
    
    return quant_t, scale, t_min


def quantize_param_dict(param_dict, bit=8, axis=-1, dither=False):
    """Helper function to quantize a parameter dictionary."""
    quantized_params = {}
    scales = {}
    t_min_vals = {}

    for k, v in param_dict.items():
        if v is None: # 'embed' key might have None value
            continue
        
        quant_v, scale, t_min = quantize_per_tensor(
            v.detach(), bit=bit, axis=axis, dither=dither
        )
        quantized_params[k] = quant_v
        scales[k] = scale
        t_min_vals[k] = t_min

    return quantized_params, scales, t_min_vals


def dequantize_param_dict(quantized_params, scales, t_min_vals, device=None):
    """Helper function to dequantize a parameter dictionary."""
    dequantized_params = {}

    for k, quantized_param in quantized_params.items():
        scale = scales[k].clone().detach().to(torch.float32)
        t_min = t_min_vals[k].clone().detach().to(torch.float32)

        # Dequantize each parameter
        dequantized_param = t_min + (scale * quantized_param)

        dequantized_param = dequantized_param.to(torch.float32)
        if device is not None:
            dequantized_param = dequantized_param.to(device)

        dequantized_params[k] = dequantized_param

    return dequantized_params

def compute_best_quant_axis(x, thres=0.05):
    """
    Compute the best quantization axis for a tensor. 
    Similar to the one used in HNeRV quantization: https://github.com/haochen-rye/HNeRV/blob/main/hnerv_utils.py#L26
    """
    best_axis = None
    best_axis_dim = 0
    for axis in range(x.ndim):
        dim = x.shape[axis]
        if x.numel() / dim >= x.numel() * thres:
            continue
        if dim > best_axis_dim:
            best_axis = axis
            best_axis_dim = dim
    return best_axis