import os
import time
from collections import defaultdict
from copy import deepcopy
from math import cos, pi

import einops
import numpy as np
import torch
from pytorch_msssim import ssim
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.io import write_video
from torchvision.utils import save_image
from tqdm import tqdm

import utils
from datasets import make
from trainers import register
from utils import make_coord_grid
from utils.compression import (
    compress_tensor,
    compress_tensor_huffman,
    decompress_tensor,
)
from utils.quantize import quantize_per_tensor

from .base_trainer import BaseTrainer


@register("nerv_enc_trainer_full_res")
class NeRVEncTrainerFullRes(BaseTrainer):

    def make_datasets(self):
        super().make_datasets()

        def get_vislist(dataset, n_vis=32):
            ids = torch.arange(n_vis) * (len(dataset) // n_vis)
            return Subset(dataset, ids.tolist())

        if hasattr(self, "train_loader"):
            self.vislist_train = get_vislist(self.train_loader.dataset)
        if hasattr(self, "test_loader_dict"):
            vislist_test = []
            for k, test_loader in self.test_loader_dict.items():
                vislist_test.append(get_vislist(test_loader.dataset))
            self.vislist_test = ConcatDataset(vislist_test)

    def adjust_learning_rate(self):
        base_lr = self.cfg["optimizer"]["args"]["lr"]
        lr_type = self.cfg["optimizer"]["lr_type"]
        max_epoch = self.cfg["max_epoch"]
        if lr_type == "cosine":
            warmup_epoch = 1
            if self.epoch <= warmup_epoch:
                lr_mult = 0.25 + 0.75 * self.epoch / warmup_epoch
            else:
                lr_mult = 0.5 * (
                    cos(pi * (self.epoch - warmup_epoch) /
                        (max_epoch - warmup_epoch))
                    + 1
                )
        elif lr_type == "step":
            lr_steps = np.array([0.9, 0.98]) * max_epoch
            lr_mult = 0.1 ** (sum(self.epoch >= lr_steps))
        else:
            NotImplementedError

        lr = base_lr * lr_mult
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self.log_temp_scalar("lr", lr)

    def train(self):
        """
            For epochs, perform training, evaluation, and visualization.
            Note that ave_scalars update ignores the actual current batch_size.
        """
        cfg = self.cfg

        self.optimizer = utils.make_optimizer(
            self.model_ddp.parameters(), cfg['optimizer'])

        max_epoch = cfg['max_epoch']
        eval_epoch = cfg.get('eval_epoch', max_epoch + 1)
        vis_epoch = cfg.get('vis_epoch', max_epoch + 1)
        save_epoch = cfg.get('save_epoch', max_epoch + 1)
        epoch_timer = utils.EpochTimer(max_epoch)

        for epoch in range(self.starting_epoch, max_epoch + 1):
            self.epoch = epoch
            self.log_buffer = [f'Epoch {epoch}']

            if self.distributed:
                for sampler in self.dist_samplers:
                    sampler.set_epoch(epoch)

            self.adjust_learning_rate()

            self.t_data, self.t_model = 0, 0
            self.train_epoch()

            if epoch % eval_epoch == 0:
                self.evaluate_epoch()

            if epoch % vis_epoch == 0:
                self.visualize_epoch()

            if epoch % save_epoch == 0:
                self.save_checkpoint(f'epoch-{epoch}.pth')
            self.save_checkpoint('epoch-last.pth')

            epoch_time, tot_time, est_time = epoch_timer.epoch_done()
            t_data_ratio = self.t_data / (self.t_data + self.t_model)
            self.log_buffer.append(
                f'{epoch_time} (d {t_data_ratio:.2f}) {tot_time}/{est_time}')
            self.log(', '.join(self.log_buffer))

        self.dump_csv(cfg)

    def _iter_step(self, data, is_train, quant_bit=32):
        vid_name_list = [name for name in data.pop("name")]
        if "metadata" in data:
            data.pop("metadata")
        data = {k: v.cuda() if isinstance(v, torch.Tensor)
                else v for k, v in data.items()}
        gt = data.pop("gt")  # b c t h w
        B = gt.shape[0]
        if not is_train:
            frames_num = self.cfg["test_dataset"]["frames"]
            input_frames, out_frames = frames_num["input"], frames_num["output"]
            if input_frames != "none":
                input_frames = [int(x) for x in input_frames.split("_")]
                data["inp"] = data["inp"][:, :, input_frames]
            if out_frames != "none":
                out_frames = [int(x) for x in out_frames.split("_")]
                gt = gt[:, :, out_frames]
        if self.cfg["generate_from_single_frame"]:
            data["inp"] = data["inp"][:, :, :1]

        start = time.time()
        model_input = {"data": data}
        if quant_bit < 32:
            model_input["quant_bit"] = quant_bit
        output = self.model_ddp(**model_input)
        enc_fps = B / (time.time() - start)

        if isinstance(output, dict):
            hyponet = output["hyponet"]
            if "hyponet_bits" in output:
                hyponet_bits = output["hyponet_bits"]
                quant_overhead_bits = output["quant_overhead_bits"]
        else:
            hyponet = output  # forward of HyperNeRV returns hyponet with the params set
        start = time.time()
        coord = make_coord_grid(gt.shape[2:3], (-1, 1), device=gt.device)
        coord = einops.repeat(coord, "t d -> b t d", b=B)

        pred = hyponet(coord)  # t b 3 h w
        torch.cuda.synchronize()
        dec_fps = B / (time.time() - start)

        gt = einops.rearrange(gt, "b c t h w -> t b c h w")
        mses = ((pred - gt) ** 2).view(B, -1).mean(dim=-1)
        loss = mses.mean()
        psnr = (-10 * torch.log10(mses)).mean()
        ssim_v = ssim(pred.flatten(end_dim=1),
                      gt.flatten(end_dim=1), data_range=1)

        return_dict = {
            "loss": loss.item(),
            "psnr": psnr.item(),
            "ssim": ssim_v.item(),
            "enc_fps": enc_fps,
            "dec_fps": dec_fps,
            "hyponet_bpp": hyponet_bits / pred.shape[0] / pred.shape[-1] / pred.shape[-2],
            "quant_overhead_bpp": quant_overhead_bits / pred.shape[0] / pred.shape[-1] / pred.shape[-2],
        }

        return_dict['total_loss'] = loss.item()

        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        else:
            if self.cfg["dump_ckt"] not in ["no"]:
                root_ckt_path = self.cfg["dump_path"]
                for batch_id, vid_name in enumerate(vid_name_list):
                    new_ckt = {
                        k: v[batch_id] for k, v in hyponet.params.items() if v != None
                    }
                    ckt_path = os.path.join(root_ckt_path, f"{vid_name}.pth")
                    torch.save(new_ckt, ckt_path)

        # Compute bpp for test set (clip- or video-level, as specified by Dataset)
        if not is_train:
            base_params = (
                self.model_ddp.module.base_params
                if hasattr(self.model_ddp, "module")
                else self.model_ddp.base_params
            )

            base_params_bits = utils.params_size_in_bits(base_params)

            first_elem_id = 0
            specific_params = {k: v[first_elem_id]
                               for k, v in hyponet.params.items() if v != None}

            specific_params_bits = utils.params_size_in_bits(specific_params)

            bpp = utils.compute_bpp(
                base_params_size_in_bits=base_params_bits,
                specific_params_size_in_bits=specific_params_bits,
                # not crop_size
                side_length=self.cfg["test_dataset"]["args"]["tubelet_size"],
                frame_num=self.cfg["test_dataset"]["args"]["frame_num"],
                pred_level="video",
            )
            return_dict["bpp"] = bpp

            # Compute bpp without base_params for comparison
            bpp_no_base = utils.compute_bpp(
                base_params_size_in_bits=0,
                specific_params_size_in_bits=specific_params_bits,
                # not crop_size
                side_length=self.cfg["test_dataset"]["args"]["tubelet_size"],
                frame_num=self.cfg["test_dataset"]["args"]["frame_num"],
                pred_level="video",
            )
            return_dict["bpp_no_base"] = bpp_no_base

        return return_dict

    def train_step(self, data):
        return self._iter_step(data, is_train=True)

    def evaluate_step(self, data, quant_bit=32):
        with torch.no_grad():
            return self._iter_step(data, is_train=False, quant_bit=quant_bit)

    def _gen_vis_result(self, tag, vislist):
        pred_value_range = (0, 1)
        self.model_ddp.eval()
        out_dir = self.cfg["env"]["save_dir"]
        res = []
        for batch_id, data in enumerate(vislist):
            data = {k: data[k].unsqueeze(0).cuda() for k in ["gt", "inp"]}
            gt = data.pop("gt")[0]
            if self.cfg["generate_from_single_frame"]:
                data["inp"] = data["inp"][:, :, :1]
            with torch.no_grad():
                model_input = {"data": data}
                output = self.model_ddp(**model_input)

                if isinstance(output, dict):
                    hyponet = output["hyponet"]
                else:
                    hyponet = output
                coord = make_coord_grid(
                    gt.shape[1:2], [-1, 1], device=gt.device)
                pred = hyponet(coord.unsqueeze(0))[:, 0]
                pred = pred.clamp(*pred_value_range)
            res.append(gt.permute(1, 0, 2, 3))
            res.append(pred)
            if self.cfg["dump_video"] not in ["no"]:
                out_vid = os.path.join(
                    out_dir, f"{batch_id:02d}_pred_{tag}.mp4")
                write_video(
                    out_vid,
                    pred.cpu().permute(0, 2, 3, 1) * 255.0,
                    fps=2,
                    options={"crf": "15"},
                )
        res = torch.stack(res)
        res = res.detach().cpu()
        if self.cfg["dump_pred"] not in ["no"]:
            self.log(f"dumped to {out_dir}")
            for i in range(res.size(0)):
                batch_id, statues = i // 2, "gt" if i % 2 == 0 else "pred"
                out_path = os.path.join(
                    out_dir, f"{batch_id:02d}_{statues}_{tag}.png")
                save_image(res[i], out_path)

    def visualize_epoch(self):
        if hasattr(self, "vislist_train"):
            self._gen_vis_result("vis_train_dataset", self.vislist_train)
        if hasattr(self, "vislist_test"):
            self._gen_vis_result("vis_test_dataset", self.vislist_test)

    def _combine_param_dict_batches(self, x_dict_list):
        """
        Combine x_dict results from multiple batches
        x_dict_list is a list of x_dicts, each of which is key -> B,N,D value
        """
        if not x_dict_list:
            return {}
        combined = {}
        for key in x_dict_list[0].keys():
            if x_dict_list[0][key] is not None:
                combined[key] = torch.cat([x_dict[key] for x_dict in x_dict_list if x_dict[key] is not None], dim=0)
            else:
                combined[key] = None
        return combined

    def _process_patches_batch(self, patches):
        """Helper to process a batch of patches through the model and get predictions"""
        patches_batch = torch.stack(patches).cuda()
        data = {'inp': patches_batch, 'gt': patches_batch}

        with torch.no_grad():
            output = self.model_ddp(data)
            return output.get('pre_mod', output.get('x_dict')), output.get('hyponet')

    def _tile_clip_from_patches(self, pred_patches, positions, metadata):
        """
        Reconstruct full clip frames from tubelet patches
        Args:
            pred_patches: Tensor of shape (T, num_patches, C, h, w)
            positions: List of frame lists, each containing patch positions
            metadata: Dataset metadata containing frame size info
        Returns:
            Reconstructed clip tensor (T,C,H,W)
        """
        crop_h = metadata['crop_size'][0].item()
        crop_w = metadata['crop_size'][1].item()

        # List of length num_patches. Same across frames.
        patch_positions = positions[0]

        T, num_patches, C, patch_h, patch_w = pred_patches.shape
        recon_clip = torch.zeros(T, C, crop_h, crop_w).cuda()
        # For each patch position
        for patch_idx in range(num_patches):
            # Get position tensors and extract values
            h_start = patch_positions[patch_idx][0].item()
            w_start = patch_positions[patch_idx][1].item()
            h_end = patch_positions[patch_idx][2].item()
            w_end = patch_positions[patch_idx][3].item()

            # Place patch for all frames at once using tensor indexing
            recon_clip[:, :, h_start:h_end,
                       w_start:w_end] = pred_patches[:, patch_idx]

        return recon_clip

    def _make_ordered_dataset(self, dataset, video_path):
        """Create ordered dataset with all frames for a video"""
        return make(self.cfg['eval_full_res_dataset'], args={
            'video_path': video_path,
            'frame_num': dataset.frame_num,
            'crop_size': dataset.crop_size
        })

    def _make_data_loader(self, dataset, batch_size, num_workers):
        """Create data loader with same settings as base loader"""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    def _encode_decode_and_compute_bits(self, original_tensor, encoding_type):
        """Encode, decode and compute bits for a tensor"""
        if encoding_type == 'arithmetic':
            meta, encoded_string = compress_tensor(
                original_tensor.clone().contiguous(), None, None, None)
            decoded_tensor = decompress_tensor(
                meta, None, None, encoded_string).cuda()
            bits = meta['bits_length'] * 8  # Convert bytes to bits
        elif encoding_type == 'huffman':
            encoded_data = compress_tensor_huffman(original_tensor)
            decoded_tensor = torch.tensor(encoded_data['codec'].decode(encoded_data['encoded']),
                                          device=original_tensor.device, dtype=original_tensor.dtype)
            decoded_tensor = decoded_tensor.reshape(original_tensor.shape)
            bits = encoded_data['bits_length']
        else:
            raise ValueError(f"Invalid encoding type: {encoding_type}")

        if not torch.allclose(original_tensor, decoded_tensor, rtol=1e-05, atol=1e-06):
            max_diff = (original_tensor - decoded_tensor).abs().max().item()
            raise ValueError(
                f"Encoding verification failed! Max difference: {max_diff}")

        return decoded_tensor, bits

    def _encode_tensor(self, tensor_to_encode, encoding_type):
        """Encodes a tensor using the specified method.

        Returns:
            encoding_info: Dictionary containing data needed for decoding.
                           (e.g., {'meta': ..., 'encoded_string': ...} for arithmetic)
                           (e.g., {'encoded_data': ...} for huffman)
            bits: Number of bits used for the encoded representation.
        """
        tensor_to_encode = tensor_to_encode.clone().contiguous()
        if encoding_type == 'arithmetic':
            meta, encoded_string = compress_tensor(
                tensor_to_encode, None, None, None)
            bits = meta['bits_length'] * 8  # Convert bytes to bits
            encoding_info = {'meta': meta, 'encoded_string': encoded_string}
        elif encoding_type == 'huffman':
            # compress_tensor_huffman returns a dict including codec, encoded, bits_length
            encoded_data = compress_tensor_huffman(tensor_to_encode)
            bits = encoded_data['bits_length']
            encoding_info = {'encoded_data': encoded_data}
        else:
            raise ValueError(f"Invalid encoding type: {encoding_type}")
        return encoding_info, bits

    def _decode_tensor(self, encoding_info, encoding_type, original_shape, device, dtype):
        """Decodes a tensor using the specified method and info."""
        if encoding_type == 'arithmetic':
            meta = encoding_info['meta']
            encoded_string = encoding_info['encoded_string']
            decoded_tensor = decompress_tensor(meta, None, None, encoded_string).to(
                device)  # Ensure on correct device
        elif encoding_type == 'huffman':
            encoded_data = encoding_info['encoded_data']
            # Decode using the codec stored in the encoded_data
            decoded_flat = encoded_data['codec'].decode(
                encoded_data['encoded'])
            decoded_tensor = torch.tensor(
                decoded_flat, device=device, dtype=dtype)
            decoded_tensor = decoded_tensor.reshape(original_shape)
        else:
            raise ValueError(f"Invalid encoding type: {encoding_type}")

        return decoded_tensor

    def _verify_encoding(self, original_tensor, decoded_tensor):
        """Performs the allclose check for verification."""
        if not torch.allclose(original_tensor, decoded_tensor, rtol=1e-05, atol=1e-06):
            max_diff = (original_tensor - decoded_tensor).abs().max().item()
            raise ValueError(
                f"Encoding verification failed! Max difference: {max_diff}")

    def quantize_param_dict(self, x_dict, quant_bit, axis):
        """Quantize param dict layer-wise"""
        x_dict_quant = {}
        scales = {}
        t_mins = {}
        for key in x_dict.keys():
            if x_dict[key] is not None:
                x_dict_quant[key], scales[key], t_mins[key] = quantize_per_tensor(
                    x_dict[key].clone(), bit=quant_bit, axis=axis, dither=False
                )
            else:
                x_dict_quant[key] = None
                scales[key] = None
                t_mins[key] = None
        return x_dict_quant, scales, t_mins

    def _process_direct(self, cur_dict, quant_bit, quant_axis, encoding_type=None):
        """Process direct dictionary with optional quantization and encoding.
        Returns:
            dict_recon: Reconstructed dictionary
            total_bits: Dictionary containing quant, overhead, and encoded bits
            time_encode: Time spent on quantization + entropy encoding for this patch
            time_decode: Time spent on entropy decoding + dequantization for this patch
        """
        if encoding_type is not None and quant_bit >= 32:
            raise ValueError("Encoding is not supported without quantization")

        t_quant, t_encode, t_decode, t_dequant = 0, 0, 0, 0
        total_bits = {'quant': 0, 'overhead': 0, 'encoded': 0}

        # Step 1: Optional Quantization (Part of Encoding)
        start_quant = time.time()
        if quant_bit < 32:
            processed_dict, scales, t_mins = self.quantize_param_dict(
                cur_dict, quant_bit, quant_axis)
            total_bits['quant'] = sum(
                v.numel() * quant_bit for v in processed_dict.values() if v is not None)
            total_bits['overhead'] = sum(
                (s.numel() + t.numel()) * 32 for s, t in zip(scales.values(), t_mins.values()) if s is not None)
        else:
            processed_dict = cur_dict  # Use original dict if no quantization
            scales = t_mins = None
            total_bits['quant'] = sum(
                v.numel() * 32 for v in cur_dict.values() if v is not None)
        t_quant = time.time() - start_quant

        # Step 2: Optional Encoding (Part of Encoding)
        start_encode = time.time()
        encoded_infos = {}  # Store encoding info per key
        # Store the state before encoding for verification
        processed_dict_quantized_state = {}
        if encoding_type:
            for key, value in processed_dict.items():
                if value is not None:
                    # Store pre-encoded state
                    processed_dict_quantized_state[key] = value.clone()
                    encoding_info, bits = self._encode_tensor(
                        value, encoding_type)
                    encoded_infos[key] = encoding_info
                    total_bits['encoded'] += bits
                else:
                    encoded_infos[key] = None
                    processed_dict_quantized_state[key] = None
        else:
            # If no encoding, store the potentially quantized state directly for verification
            processed_dict_quantized_state = processed_dict
        t_encode = time.time() - start_encode

        # Step 3: Optional Entropy Decoding (Part of Decoding)
        start_decode = time.time()
        decoded_dict_for_dequant = {}
        if encoding_type:
            for key, value in processed_dict.items():
                if value is not None:
                    decoded_tensor = self._decode_tensor(
                        encoded_infos[key], encoding_type, value.shape, value.device, value.dtype)
                    decoded_dict_for_dequant[key] = decoded_tensor
                else:
                    decoded_dict_for_dequant[key] = None
        else:
            # If no encoding was done, the dict for dequant is the potentially quantized dict
            decoded_dict_for_dequant = processed_dict
        t_decode = time.time() - start_decode

        # Verification Step (Run AFTER decoding, outside timing)
        if encoding_type:
            for key, decoded_tensor in decoded_dict_for_dequant.items():
                if decoded_tensor is not None:
                    self._verify_encoding(
                        processed_dict_quantized_state[key], decoded_tensor)

        # Step 4: Dequantize (Part of Decoding)
        start_dequant = time.time()
        if quant_bit < 32:
            # Pass None as base_dict since this is direct processing, not residual
            dict_recon = self.recover_param_dict_from_quantized_residuals(
                None, decoded_dict_for_dequant, scales, t_mins)
        else:
            # If no quant, this is original or decoded(if encoded)
            dict_recon = decoded_dict_for_dequant
        t_dequant = time.time() - start_dequant

        time_encode_patch = t_quant + t_encode
        time_decode_patch = t_decode + t_dequant

        return dict_recon, total_bits, time_encode_patch, time_decode_patch

    def compute_param_dict_residuals(self, x_dict_current, x_dict_base):
        """Compute residuals between two param dicts layer-wise"""
        residuals = {}
        for key in x_dict_current.keys():
            if x_dict_current[key] is not None:
                residuals[key] = x_dict_current[key] - x_dict_base[key]
            else:
                residuals[key] = None
        return residuals

    def _process_residual(self, cur_dict, base_dict, quant_bit, quant_axis, encoding_type=None):
        """Process residuals and return reconstruction and bits.
        Returns:
            dict_recon: Reconstructed dictionary using residuals
            encoded_bits: Bits used for the *encoded residual*
            time_encode: Time for residual calc + quant + entropy encoding
            time_decode: Time for entropy decoding + dequant + adding base
        """
        if encoding_type is not None and quant_bit >= 32:
            raise ValueError("Encoding is not supported without quantization")

        t_residual_calc, t_quant, t_encode, t_decode, t_dequant_add = 0, 0, 0, 0, 0
        encoded_bits = 0

        # Calculate Residual (Part of Encoding)
        start_residual_calc = time.time()
        residual = self.compute_param_dict_residuals(cur_dict, base_dict)
        t_residual_calc = time.time() - start_residual_calc

        # Step 1: Optional Quantization of Residual (Part of Encoding)
        start_quant = time.time()
        if quant_bit < 32:
            processed_residual, scales, t_mins = self.quantize_param_dict(
                residual, quant_bit, quant_axis)
        else:
            processed_residual = residual
            scales = t_mins = None
        t_quant = time.time() - start_quant

        # Step 2: Optional Encoding of Residual (Part of Encoding)
        start_encode = time.time()
        encoded_infos = {}
        # Store pre-encoded state for verification
        processed_residual_quantized_state = {}
        if encoding_type:
            for key, value in processed_residual.items():
                if value is not None:
                    processed_residual_quantized_state[key] = value.clone()
                    encoding_info, bits = self._encode_tensor(
                        value, encoding_type)
                    encoded_infos[key] = encoding_info
                    encoded_bits += bits
                else:
                    encoded_infos[key] = None
                    processed_residual_quantized_state[key] = None
        else:
            processed_residual_quantized_state = processed_residual
        t_encode = time.time() - start_encode

        # Step 3: Optional Entropy Decoding of Residual (Part of Decoding)
        start_decode = time.time()
        decoded_residual_for_dequant = {}
        if encoding_type:
            for key, value in processed_residual.items():
                if value is not None:
                    decoded_tensor = self._decode_tensor(
                        encoded_infos[key], encoding_type, value.shape, value.device, value.dtype)
                    decoded_residual_for_dequant[key] = decoded_tensor
                else:
                    decoded_residual_for_dequant[key] = None
        else:
            decoded_residual_for_dequant = processed_residual
        t_decode = time.time() - start_decode

        # Verification Step (Run after decoding, outside timing)
        if encoding_type:
            for key, decoded_tensor in decoded_residual_for_dequant.items():
                if decoded_tensor is not None:
                    self._verify_encoding(
                        processed_residual_quantized_state[key], decoded_tensor)

        # Step 4: Dequantize Residual and Add Base (Part of Decoding)
        start_dequant_add = time.time()
        if quant_bit < 32:
            # Dequantize the decoded residual and add base
            dict_recon = self.recover_param_dict_from_quantized_residuals(
                base_dict, decoded_residual_for_dequant, scales, t_mins)
        else:
            # If no quantization, add the decoded (or raw if no encoding) residual to base
            dict_recon = {}
            for key, res_value in decoded_residual_for_dequant.items():
                if res_value is not None:
                    # Ensure base_dict exists and has the key before adding
                    if base_dict is not None and key in base_dict and base_dict[key] is not None:
                        dict_recon[key] = base_dict[key] + res_value
                    else:
                        # Handle case where base might be missing or None (e.g., if cloning failed)
                        # Or if residual exists but base doesn't (unlikely but possible)
                        dict_recon[key] = res_value
                else:
                    dict_recon[key] = None  # No base and no residual

        t_dequant_add = time.time() - start_dequant_add

        time_encode_patch = t_residual_calc + t_quant + t_encode
        time_decode_patch = t_decode + t_dequant_add

        return dict_recon, encoded_bits, time_encode_patch, time_decode_patch

    def _log_full_res_metrics(self, recon_type, dataset_name, csv_path, csv_path_per_video, metrics_per_video, clip_metrics_across_videos,
                              ordered_dataset, videos, encoding_type, quant_bit, quant_axis, log_per_video=True):
        """Log full resolution metrics for the dataset"""
        patch_h, patch_w = ordered_dataset.tubelet_size
        pixels_per_clip = patch_h * patch_w * ordered_dataset.frame_num

        self.log(f'\nFull-res {recon_type} residuals eval on {dataset_name}: '
                 f'quant_bit={quant_bit}, quant_axis={quant_axis}, encoding={encoding_type or "none"}')

        if log_per_video:
            for video in videos:
                video_metrics = metrics_per_video[video]
                log_buffer = [f'\nVideo: {os.path.basename(video)}']

                bpp_quant = np.mean(
                    video_metrics['bits_quant']) / pixels_per_clip
                overhead_bpp_quant = np.mean(
                    video_metrics['bits_quant_overhead']) / pixels_per_clip

                for method in ['direct', 'from_first', 'from_prev']:
                    avg_psnr = np.mean(video_metrics[f'{method}_psnr'])
                    avg_ssim = np.mean(video_metrics[f'{method}_ssim'])

                    # Calculate encoded bpp by taking mean first
                    bpp_encoded = np.mean(
                        video_metrics[f'{method}_bits_encoded']) / pixels_per_clip if encoding_type else 0
                    # Final bpp is encoded if available, otherwise quant + overhead bpp
                    bpp_total = bpp_encoded if encoding_type else (
                        bpp_quant + overhead_bpp_quant)

                    avg_enc_fps = np.mean(video_metrics[f'{method}_enc_fps'])
                    avg_dec_fps = np.mean(video_metrics[f'{method}_dec_fps'])

                    log_buffer.extend([
                        f'\n{recon_type}_recon - {method}:',
                        f'avg_psnr={avg_psnr:.4f}',
                        f'avg_ssim={avg_ssim:.4f}',
                        f'bpp_quant={bpp_quant:.4f}',
                        f'bpp_quant_with_overhead={(bpp_quant + overhead_bpp_quant):.4f}',
                        f'bpp_encoded={bpp_encoded:.4f}',
                        f'bpp_total={bpp_total:.4f}',
                        f'enc_fps={avg_enc_fps:.2f}',
                        f'dec_fps={avg_dec_fps:.2f}'
                    ])

                self.log(', '.join(log_buffer))

        self.log('\nAverages across all videos:')

        avg_bpp_quant = np.mean(
            clip_metrics_across_videos['bits_quant']) / pixels_per_clip
        avg_overhead_bpp_quant = np.mean(
            clip_metrics_across_videos['bits_quant_overhead']) / pixels_per_clip

        for method in ['direct', 'from_first', 'from_prev']:
            avg_psnr = np.mean(clip_metrics_across_videos[method]['psnr'])
            avg_ssim = np.mean(clip_metrics_across_videos[method]['ssim'])

            avg_bpp_encoded = np.mean(
                clip_metrics_across_videos[method]['bits_encoded']) / pixels_per_clip if encoding_type else 0
            avg_bpp_total = avg_bpp_encoded if encoding_type else (
                avg_bpp_quant + avg_overhead_bpp_quant)

            avg_enc_fps = np.mean(
                clip_metrics_across_videos[method]['enc_fps'])
            avg_dec_fps = np.mean(
                clip_metrics_across_videos[method]['dec_fps'])

            self.log(
                f'\n{recon_type}_recon - {method} (average): '
                f'avg_psnr={avg_psnr:.4f}, '
                f'avg_ssim={avg_ssim:.4f}, '
                f'bpp_quant={avg_bpp_quant:.4f}, '
                f'bpp_quant_with_overhead={(avg_bpp_quant + avg_overhead_bpp_quant):.4f}, '
                f'bpp_encoded={avg_bpp_encoded:.4f}, '
                f'bpp_total={avg_bpp_total:.4f}, '
                f'enc_fps={avg_enc_fps:.2f}, '
                f'dec_fps={avg_dec_fps:.2f}'
            )

        if csv_path:
            # Save full resolution evaluation metrics to CSV
            csv_columns = [
                'quant_bit',
                'residual_type',
                'psnr',
                'ssim',
                'bpp_total',
                'enc_fps',
                'dec_fps',
                'video',
                'reconstruction_type',
                'quant_axis',
                'encoding_type',
                'bpp_quant',
                'bpp_quant_with_overhead',
                'bpp_encoded',
            ]
            if log_per_video:
                file_exists = os.path.isfile(csv_path_per_video)
                mode = 'a' if file_exists else 'w'
                with open(csv_path_per_video, mode) as f:
                    if not file_exists:
                        f.write(','.join(csv_columns) + '\n')

                    for video in videos:
                        video_metrics = metrics_per_video[video]
                        video_name = os.path.basename(video)

                        for method in ['direct', 'from_first', 'from_prev']:
                            avg_psnr = np.mean(video_metrics[f'{method}_psnr'])
                            avg_ssim = np.mean(video_metrics[f'{method}_ssim'])

                            bpp_encoded = np.mean(
                                video_metrics[f'{method}_bits_encoded']) / pixels_per_clip if encoding_type else 0
                            # Final bpp is encoded bpp, if enabled, otherwise quant + overhead bpp
                            bpp_total = bpp_encoded if encoding_type else (
                                bpp_quant + overhead_bpp_quant)

                            avg_enc_fps = np.mean(
                                video_metrics[f'{method}_enc_fps'])
                            avg_dec_fps = np.mean(
                                video_metrics[f'{method}_dec_fps'])

                            row = [
                                f'{quant_bit}',
                                method,
                                f'{avg_psnr:.4f}',
                                f'{avg_ssim:.4f}',
                                f'{bpp_total:.4f}',
                                f'{avg_enc_fps:.2f}',
                                f'{avg_dec_fps:.2f}',
                                video_name,
                                recon_type,
                                f'{quant_axis}',
                                f'{encoding_type or "none"}',
                                f'{bpp_quant:.4f}',
                                f'{bpp_quant + overhead_bpp_quant:.4f}',
                                f'{bpp_encoded:.4f}',
                            ]
                            f.write(','.join(row) + '\n')

            file_exists = os.path.isfile(csv_path)
            with open(csv_path, 'a' if file_exists else 'w') as f:
                if not file_exists:
                    f.write(','.join(csv_columns) + '\n')
                # Write averages across all videoss
                for method in ['direct', 'from_first', 'from_prev']:
                    avg_psnr = np.mean(
                        clip_metrics_across_videos[method]['psnr'])
                    avg_ssim = np.mean(
                        clip_metrics_across_videos[method]['ssim'])

                    avg_bpp_encoded = np.mean(
                        clip_metrics_across_videos[method]['bits_encoded']) / pixels_per_clip if encoding_type else 0
                    avg_bpp_total = avg_bpp_encoded if encoding_type else (
                        avg_bpp_quant + avg_overhead_bpp_quant)

                    avg_enc_fps = np.mean(
                        clip_metrics_across_videos[method]['enc_fps'])
                    avg_dec_fps = np.mean(
                        clip_metrics_across_videos[method]['dec_fps'])

                    row = [
                        f'{quant_bit}',
                        method,
                        f'{avg_psnr:.4f}',
                        f'{avg_ssim:.4f}',
                        f'{avg_bpp_total:.4f}',
                        f'{avg_enc_fps:.2f}',
                        f'{avg_dec_fps:.2f}',
                        'all',
                        recon_type,
                        f'{quant_axis}',
                        f'{encoding_type or "none"}',
                        f'{avg_bpp_quant:.4f}',
                        f'{avg_bpp_quant + avg_overhead_bpp_quant:.4f}',
                        f'{avg_bpp_encoded:.4f}',
                    ]
                    f.write(','.join(row) + '\n')
            self.log(
                f'Full-res residuals eval {dataset_name} saved to {csv_path} and {csv_path_per_video}')

    def recover_param_dict_from_quantized_residuals(self, base_dict, processed_dict, scales, t_mins):
        """
        Reconstruct param dict from quantized residuals
        Args:
            base_dict: Base parameter dictionary or None for direct reconstruction
            processed_dict: Quantized and encoded-decoded dictionary
            scales: Quantization scales dictionary
            t_mins: Quantization minimums dictionary

        Returns:
            dict_recon: Reconstructed parameter dictionary

        """
        dict_recon = {}
        for key in processed_dict.keys():
            if processed_dict[key] is not None:
                dequant_val = t_mins[key] + (scales[key] * processed_dict[key])
                if base_dict is not None:
                    dict_recon[key] = base_dict[key] + dequant_val
                else:
                    # not a residual
                    dict_recon[key] = dequant_val
            else:
                if base_dict is not None:
                    dict_recon[key] = base_dict[key]
                else:
                    dict_recon[key] = None  # wb3
        return dict_recon

    def convert_x_dict_to_params(self, B, x_dict, base_params, model):
        """Helper function to compute parameters from x_dict"""
        params = {'embed': None}
        for name, shape in model.hyponet.param_shapes.items():
            wb = einops.repeat(base_params[name], 'n m -> b n m', b=B)
            init_w, init_b = wb[:, :-1, :], wb[:, -1:, :]

            if x_dict[name] is not None:
                x = x_dict[name]

                repeat_num = init_w.nelement() // x.nelement()
                # (B, in_ch, out_ch)
                x = einops.repeat(x, 'B n m -> B n d m',
                                  d=repeat_num).reshape_as(init_w)
                assert x.shape[0] == B
                w = F.normalize(init_w * x, dim=1)
            else:
                w = F.normalize(init_w, dim=1)

            wb = torch.cat([w, init_b], dim=1)
            params[name] = wb

        return params

    def _init_clip_metrics(self):
        """Initialize metrics dictionary for a clip"""
        # Common metrics across all types
        metrics = {
            'bits_quant': 0,
            'bits_quant_overhead': 0,
        }

        base = {
            'psnr': [],
            'ssim': [],
            'bits_encoded': [],
            'enc_fps': [],
            'dec_fps': [],
        }
        metrics.update({k: deepcopy(base)
                       for k in ['from_first', 'from_prev', 'direct']})

        return metrics

    def _init_clip_metrics_across_videos(self):
        """Initialize metrics dictionary for a clip"""
        # Common metrics across all types
        metrics = {
            'bits_quant': [],
            'bits_quant_overhead': [],
        }

        base = {
            'psnr': [],
            'ssim': [],
            'bits_encoded': [],
            'enc_fps': [],
            'dec_fps': [],
        }
        metrics.update({k: deepcopy(base)
                       for k in ['from_first', 'from_prev', 'direct']})

        return metrics

    def _clone_dict(self, x_dict):
        """Deep clone a dictionary of tensors"""
        return {k: v.clone() if v is not None else None for k, v in x_dict.items()}

    def _combine_batch_reconstructions(self, recon_batch_list):
        """Combine batch reconstructions"""
        return {
            k: torch.cat([d[k] for d in recon_batch_list], dim=0)
            if recon_batch_list[0][k] is not None else None
            for k in recon_batch_list[0].keys()
        }

    def _compute_full_clip_metrics(self, metrics, recon_clip, gt_clip):
        """Update metrics for a clip"""

        gt_clip = einops.rearrange(gt_clip, "c t h w -> t c h w")
        mse = ((recon_clip - gt_clip) ** 2).mean()
        psnr = -10 * torch.log10(mse)
        ssim_v = ssim(recon_clip, gt_clip, data_range=1)

        metrics['psnr'].append(psnr.item())
        metrics['ssim'].append(ssim_v.item())
        return metrics

    def _update_clip_metrics_across_videos(self, clip_metrics_across_videos, clip_metrics):
        """Update metrics across all videos"""
        if clip_metrics['bits_quant'] > 0:
            clip_metrics_across_videos['bits_quant'].append(
                clip_metrics['bits_quant'])
            clip_metrics_across_videos['bits_quant_overhead'].append(
                clip_metrics['bits_quant_overhead'])
        for method in ['direct', 'from_first', 'from_prev']:
            for metric, value in clip_metrics[method].items():
                if isinstance(value, list):  # psnr, ssim, bits_encoded
                    clip_metrics_across_videos[method][metric].extend(value)
                else:
                    clip_metrics_across_videos[method][metric] = value
        return clip_metrics_across_videos

    def reconstruct_from_x_dict(self, B, T, x_dict_recon, device):
        # Reconstruct from quantized then dequantized x_dict
        coord = make_coord_grid((T,), (-1, 1), device=device)
        coord = einops.repeat(coord, "t d -> b t d", b=B)
        if hasattr(self.model_ddp, "module"):
            base_params = self.model_ddp.module.base_params
            model = self.model_ddp.module
        else:
            base_params = self.model_ddp.base_params
            model = self.model_ddp
        # Reconstruct using quantized x_dict
        params = self.convert_x_dict_to_params(
            B, x_dict_recon, base_params, model)
        hyponet_x = model.hyponet
        hyponet_x.set_params(params)
        pred_x = hyponet_x(coord)
        return pred_x

    def reconstruct_from_weights(self, B, T, weights, device):
        """Reconstruct output using provided weights"""
        coord = make_coord_grid((T,), (-1, 1), device=device)
        coord = einops.repeat(coord, "t d -> b t d", b=B)

        model = self.model_ddp.module if hasattr(
            self.model_ddp, "module") else self.model_ddp

        hyponet_w = model.hyponet
        hyponet_w.set_params(weights)
        pred_w = hyponet_w(coord)
        return pred_w

    def evaluate_x_dict_residuals_epoch(self, quant_bit=8, quant_axis=0,
                                        encoding_type='arithmetic', eval_csv_prefix='', log_per_video=True, chunk_pred_batch_size=None):
        """
        Evaluate model performance with x_dict residuals and quantization at full resolution
        using patch-based processing.
        """
        if encoding_type not in [None, 'arithmetic', 'huffman']:
            raise ValueError(f"Invalid encoding_type: {encoding_type}")

        self.model_ddp.eval()
        metrics_per_video = {}

        # can be extended to evaluate multiple datasets
        dataset_name, loader = next(iter(self.test_loader_dict.items()))
        dataset = loader.dataset
        videos = dataset.vid_list if hasattr(dataset, 'vid_list') else []
        clip_metrics_across_videos = self._init_clip_metrics_across_videos()
        self.log(f'Full-res x_dict residuals eval: Processing {dataset_name}')
        for video in tqdm(videos):
            ordered_dataset = self._make_ordered_dataset(dataset, video)
            ordered_loader = self._make_data_loader(
                ordered_dataset, batch_size=1, num_workers=loader.num_workers)

            # Stores the reconstructed x_dict of the first clip and
            # from the previous clip for each position
            first_x_dicts_recon_per_pos = {}
            prev_x_dicts_recon_per_pos = {}

            # Reset video metrics for each video
            video_metrics = defaultdict(list)

            for data in ordered_loader:
                # Remove batch dimension since we process one clip at a time
                data = {k: v[0].cuda() if isinstance(v, torch.Tensor) else v
                        for k, v in data.items()}  # unbatchify inp, gt tensors
                start_time = time.time()
                patches = data['patches']
                # len = num_patches in frame.
                positions = data['patch_positions']
                start_frame = data['start_frame']  # single value

                clip_metrics = self._init_clip_metrics()

                # Process patches for all frames in clip together
                start_common_encode = time.time()
                all_patches = [[] for _ in range(len(patches[0]))]
                for frame_patches in patches:  # frame_patches is list of 12 patches
                    for i, patch in enumerate(frame_patches):
                        all_patches[i].append(patch.squeeze(0))
                all_patches = [torch.stack(patch_seq, dim=1)
                               for patch_seq in all_patches]
                x_dict_all, _ = self._process_patches_batch(all_patches)
                # Time to get raw x_dict for clip
                t_common_encode = time.time() - start_common_encode

                clip_metrics = self._init_clip_metrics()
                x_dict_recon_patches = {'direct': [],
                                        'from_first': [], 'from_prev': []}

                # Initialize clip time accumulators
                clip_t_compress = {
                    'encode': {'direct': 0, 'from_first': 0, 'from_prev': 0},
                    'decode': {'direct': 0, 'from_first': 0, 'from_prev': 0}
                }

                current_clip_bits_quant = []
                current_clip_bits_overhead = []
                current_clip_bits_encoded_direct = []
                current_clip_bits_encoded_first = []
                current_clip_bits_encoded_prev = []

                # Process each patch position
                # pos_tuple (h_start, w_start, h_end, w_end)
                for pos_idx, pos_tuple in enumerate(positions[0]):
                    # Ensure pos_tuple is hashable (it should be if it contains basic types)
                    # Convert tensor elements to standard Python types if needed
                    pos_key = tuple(p.item() for p in pos_tuple)
                    cur_x_dict = {
                        # Maintain batch dim of 1
                        k: v[pos_idx:pos_idx+1].cuda()
                        if v is not None else None
                        for k, v in x_dict_all.items()
                    }

                    # Direct processing
                    x_dict_recon_direct, bits_direct, t_enc_direct, t_dec_direct = self._process_direct(
                        cur_x_dict, quant_bit, quant_axis, encoding_type)
                    x_dict_recon_patches['direct'].append(x_dict_recon_direct)
                    # Store bits for this specific patch
                    current_clip_bits_encoded_direct.append(
                        bits_direct['encoded'])
                    clip_t_compress['encode']['direct'] += t_enc_direct
                    clip_t_compress['decode']['direct'] += t_dec_direct

                    if start_frame == 0:
                        # First clip for this video: store initial states for this position

                        # Reconstructions are the same as direct for the first clip
                        x_dict_recon_first = x_dict_recon_direct
                        x_dict_recon_prev = x_dict_recon_direct

                        first_x_dicts_recon_per_pos[pos_key] = self._clone_dict(
                            x_dict_recon_first)
                        prev_x_dicts_recon_per_pos[pos_key] = self._clone_dict(
                            x_dict_recon_prev)

                        x_dict_recon_patches['from_first'].append(
                            x_dict_recon_first)
                        x_dict_recon_patches['from_prev'].append(
                            x_dict_recon_prev)
                        current_clip_bits_encoded_first.append(
                            bits_direct['encoded'])  # Same as direct
                        current_clip_bits_encoded_prev.append(
                            bits_direct['encoded'])  # Same as direct

                        clip_t_compress['encode']['from_first'] += t_enc_direct
                        clip_t_compress['decode']['from_first'] += t_dec_direct
                        clip_t_compress['encode']['from_prev'] += t_enc_direct
                        clip_t_compress['decode']['from_prev'] += t_dec_direct

                        # Store common quant/overhead bits for the first clip (per patch)
                        current_clip_bits_quant.append(bits_direct['quant'])
                        current_clip_bits_overhead.append(
                            bits_direct['overhead'])

                    else:
                        # Subsequent clips: calculate residuals using stored states for this position

                        if pos_key not in first_x_dicts_recon_per_pos or pos_key not in prev_x_dicts_recon_per_pos:
                            raise RuntimeError(
                                f"Missing state for position {pos_key}...")

                        x_dict_recon_first, enc_bits_first, t_enc_p_first, t_dec_p_first = self._process_residual(
                            cur_x_dict, first_x_dicts_recon_per_pos[pos_key], quant_bit, quant_axis,
                            encoding_type)
                        x_dict_recon_patches['from_first'].append(
                            x_dict_recon_first)
                        current_clip_bits_encoded_first.append(enc_bits_first)
                        clip_t_compress['encode']['from_first'] += t_enc_p_first
                        clip_t_compress['decode']['from_first'] += t_dec_p_first

                        x_dict_recon_prev, enc_bits_prev, t_enc_p_prev, t_dec_p_prev = self._process_residual(
                            cur_x_dict, prev_x_dicts_recon_per_pos[pos_key], quant_bit, quant_axis,
                            encoding_type)
                        x_dict_recon_patches['from_prev'].append(
                            x_dict_recon_prev)
                        current_clip_bits_encoded_prev.append(enc_bits_prev)
                        clip_t_compress['encode']['from_prev'] += t_enc_p_prev
                        clip_t_compress['decode']['from_prev'] += t_dec_p_prev

                        prev_x_dicts_recon_per_pos[pos_key] = self._clone_dict(
                            x_dict_recon_prev)

                if current_clip_bits_quant:
                    clip_metrics['bits_quant'] = np.mean(
                        current_clip_bits_quant)
                    clip_metrics['bits_quant_overhead'] = np.mean(
                        current_clip_bits_overhead)
                clip_metrics['direct']['bits_encoded'].append(np.mean(
                    current_clip_bits_encoded_direct))  # take average of tubelets within clip
                clip_metrics['from_first']['bits_encoded'].append(np.mean(
                    current_clip_bits_encoded_first))  # take average of tubelets within clip
                clip_metrics['from_prev']['bits_encoded'].append(
                    np.mean(current_clip_bits_encoded_prev))  # take average of tubelets within clip

                # Reconstruct full frames and compute PSNR/SSIM metrics for the clip
                for method in ['direct', 'from_first', 'from_prev']:
                    # Hyponet Reconstruction + Tiling (Final Decode Steps)
                    start_hyponet_tile = time.time()
                    recon_batch = self._combine_batch_reconstructions(
                        x_dict_recon_patches[method])
                    num_patches_processed = len(x_dict_recon_patches[method])
                    if num_patches_processed != len(positions[0]):
                        raise ValueError(f"Batch size mismatch for {method}")

                    recon_patches = self.reconstruct_from_x_dict(
                        num_patches_processed, dataset.frame_num, recon_batch, device=data[
                            'gt'].device
                    )
                    recon_clip = self._tile_clip_from_patches(
                        recon_patches, positions, data['metadata'])
                    t_hyponet_tile = time.time() - start_hyponet_tile

                    # Calculate Total Clip Times
                    total_encode_time_clip = t_common_encode + \
                        clip_t_compress['encode'][method]
                    total_decode_time_clip = clip_t_compress['decode'][method] + \
                        t_hyponet_tile

                    # Calculate FPS
                    clip_metrics[method]['enc_fps'].append(
                        dataset.frame_num / total_encode_time_clip)
                    clip_metrics[method]['dec_fps'].append(
                        dataset.frame_num / total_decode_time_clip)

                    # Get psnr, ssim for the clip
                    clip_metrics[method] = self._compute_full_clip_metrics(
                        clip_metrics[method], recon_clip, data['gt']
                    )

                # Aggregate clip metrics into video_metrics and clip_metrics_across_videos
                for method in ['direct', 'from_first', 'from_prev']:
                    # these are now lists with one value per clip
                    for metric in ['psnr', 'ssim']:
                        video_metrics[f'{method}_{metric}'].extend(
                            clip_metrics[method][metric])
                    for metric in ['enc_fps', 'dec_fps']:  # single element list
                        video_metrics[f'{method}_{metric}'].extend(
                            clip_metrics[method][metric])
                    video_metrics[f'{method}_bits_encoded'].extend(
                        clip_metrics[method]['bits_encoded'])  # single element list
                # Store quant/overhead bits (these are averages per clip on 'direct' method)
                video_metrics['bits_quant'].append(clip_metrics['bits_quant'])
                video_metrics['bits_quant_overhead'].append(
                    clip_metrics['bits_quant_overhead'])

                clip_metrics_across_videos = self._update_clip_metrics_across_videos(
                    clip_metrics_across_videos, clip_metrics)

            # Store video metrics
            metrics_per_video[video] = video_metrics

        if self.is_master:
            prefix = ''
            if isinstance(eval_csv_prefix, str):
                if eval_csv_prefix.strip().lower() not in ['', 'none', 'null']:
                    prefix = f'{eval_csv_prefix.strip()}_'

            csv_path = os.path.join(
                self.cfg["eval_metrics_path"],
                f'{prefix}eval_full_res_residuals_{dataset_name}.csv',
            )
            csv_path_per_video = os.path.join(
                self.cfg["eval_metrics_path"],
                f'{prefix}eval_per_vid_full_res_residuals_{dataset_name}.csv',
            )

            self._log_full_res_metrics(
                'x_dict', dataset_name, csv_path, csv_path_per_video, metrics_per_video, clip_metrics_across_videos,
                ordered_dataset, videos, encoding_type, quant_bit, quant_axis, log_per_video=log_per_video
            )

        return metrics_per_video, csv_path
