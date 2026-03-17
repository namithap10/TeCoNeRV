import logging
import os
import shutil
import time
from typing import Dict

import numpy as np
import torch
from torch._C import dtype
from torch.optim import SGD, Adam

DTYPE_BIT_SIZE: Dict[dtype, int] = {
    torch.float32: 32,
    torch.float: 32,
    torch.float64: 64,
    torch.double: 64,
    torch.float16: 16,
    torch.half: 16,
    torch.bfloat16: 16,
    torch.complex32: 32,
    torch.complex64: 64,
    torch.complex128: 128,
    torch.cdouble: 128,
    torch.uint8: 8,
    torch.int8: 8,
    torch.int16: 16,
    torch.short: 16,
    torch.int32: 32,
    torch.int: 32,
    torch.int64: 64,
    torch.long: 64,
    torch.bool: 1,
}


def ensure_path(path, replace=True):
    basename = os.path.basename(path.rstrip("/"))
    if os.path.exists(path):
        if replace and basename.startswith("_"):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def set_logger(file_path):
    logger = logging.getLogger()
    logger.setLevel("INFO")
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(file_path, "w")
    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s", "%m-%d %H:%M:%S")
    for handler in [stream_handler, file_handler]:
        handler.setFormatter(formatter)
        handler.setLevel("INFO")
        logger.addHandler(handler)
    return logger


def set_save_dir(save_dir, replace=True):
    ensure_path(save_dir, replace=replace)
    time_str = time.strftime("%Y_%m_%d_%H_%M_%S")
    logger = set_logger(os.path.join(save_dir, f"log_{time_str}.txt"))
    writer = None
    return logger, writer, time_str


def compute_num_params(model, full_model=True, text=True):
    if full_model:
        tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    else:
        tot = int(sum([v.nelement() for k, v in model.base_params.items()]))

    if text:
        if tot >= 1e6:
            return "{:.1f}M".format(tot / 1e6)
        elif tot >= 1e3:
            return "{:.1f}K".format(tot / 1e3)
        else:
            return str(tot)
    else:
        return tot


def psnr(img1, img2):
    """Calculates PSNR between two images.

    Args:
        img1 (torch.Tensor):
        img2 (torch.Tensor):
    """
    return 20. * np.log10(1.) - 10. * (img1 - img2).detach().pow(2).mean().log10().to('cpu').item()


def state_dict_size_in_bits(state_dict):
    """Calculate total number of bits to store `state_dict`."""
    return sum(
        sum(t.nelement() * DTYPE_BIT_SIZE[t.dtype] for t in tensors)
        for tensors in state_dict.values()
    )


def model_size_in_bits(model):
    """Calculate total number of bits to store `model` parameters and buffers."""
    return sum(
        sum(t.nelement() * DTYPE_BIT_SIZE[t.dtype] for t in tensors)
        for tensors in (model.parameters(), model.buffers())
    )


def params_size_in_bits(params):
    """
    Calculate number of bits in params passed of k:v form.
    params may be base params or clip/video-specific params.
    """

    params_size_in_bits = sum(
        v.nelement() * DTYPE_BIT_SIZE[v.dtype] for v in params.values() if v is not None
    )
    return params_size_in_bits


def compute_bpp(
    base_params_size_in_bits,
    specific_params_size_in_bits,
    side_length,
    frame_num,
    num_frames_total=None,
    pred_level="video",
):
    """
    When pred_level='clip', predictions are made at the clip level and we want video_bpp.
    In this case, the number of bits required to store the video is dependent on the 
    number of clips in the video. specific_params_size_in_bits is then the number
    of bits for clip-specific params.
    """

    def calculate_num_pixels(num_frames: int) -> int:
        # Handle the case where side_length is a string like "480_640" or a tuple/list like (480, 640)
        if isinstance(side_length, str) and '_' in side_length:
            dims = [int(x) for x in side_length.split('_')]
            return dims[0] * dims[1] * num_frames
        elif isinstance(side_length, (tuple, list)):
            return side_length[0] * side_length[1] * num_frames
        else:
            try:
                sl = int(side_length)
                return sl ** 2 * num_frames
            except (ValueError, TypeError):
                print(
                    f"Warning: Unsupported side_length format: {side_length}, using default 256x256")
                return 256 * 256 * num_frames

    if pred_level == "clip":
        if num_frames_total is None:
            raise ValueError(
                "num_frames_total for video must be provided for clip-level prediction")

        num_clips = num_frames_total / frame_num
        video_size_in_bits = (
            base_params_size_in_bits
            + num_clips * specific_params_size_in_bits
        )

        video_bpp = video_size_in_bits / calculate_num_pixels(num_frames_total)
        return video_bpp

    # else prediction is considered to be at the video level - length frame_num
    size_in_bits = base_params_size_in_bits + specific_params_size_in_bits
    return size_in_bits / calculate_num_pixels(frame_num)


def text2str(tot):
    if tot >= 1e6:
        return "{:.1f}M".format(tot / 1e6)
    elif tot >= 1e3:
        return "{:.1f}K".format(tot / 1e3)
    else:
        return str(tot)


def make_optimizer(params, optimizer_spec, load_sd=False):
    optimizer = {"sgd": SGD, "adam": Adam}[optimizer_spec["name"]](
        params, **optimizer_spec["args"]
    )
    if load_sd:
        optimizer.load_state_dict(optimizer_spec["sd"])
    return optimizer


class Averager:

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v

class EpochTimer:

    def __init__(self, max_epoch):
        self.max_epoch = max_epoch
        self.epoch = 0
        self.t_start = time.time()
        self.t_last = self.t_start

    def epoch_done(self):
        t_cur = time.time()
        self.epoch += 1
        epoch_time = t_cur - self.t_last
        tot_time = t_cur - self.t_start
        est_time = tot_time / self.epoch * self.max_epoch
        self.t_last = t_cur
        return time_text(epoch_time), time_text(tot_time), time_text(est_time)


def time_text(secs):
    if secs >= 3600:
        return f"{secs / 3600:.1f}h"
    elif secs >= 60:
        return f"{secs / 60:.1f}m"
    else:
        return f"{secs:.1f}s"