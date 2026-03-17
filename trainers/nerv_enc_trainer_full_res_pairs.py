import os
import time
from math import cos, pi

import einops
import numpy as np
import torch
import torch.nn as nn
from pytorch_msssim import ssim
from torch.utils.data import ConcatDataset, Subset
from torchvision.io import write_video
from torchvision.utils import save_image

import utils
from trainers import register
from utils import make_coord_grid

from .base_trainer import BaseTrainer


@register("nerv_enc_trainer_full_res_pairs")
class NeRVEncTrainerFullResPairs(BaseTrainer):

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
                    cos(pi * (self.epoch - warmup_epoch) / (max_epoch - warmup_epoch))
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

        self.optimizer = utils.make_optimizer(self.model_ddp.parameters(), cfg['optimizer'])

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
            self.log_buffer.append(f'{epoch_time} (d {t_data_ratio:.2f}) {tot_time}/{est_time}')
            self.log(', '.join(self.log_buffer))

        self.dump_csv(cfg)

    def _iter_step(self, data, is_train, quant_bit=32):
        vid_name_list = [name for name in data.pop("name")]
        if "metadata" in data:
            data.pop("metadata")
        data = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        gt = data.pop("gt") # b 2 c t h w
        B = gt.shape[0]
        
        # GT for first clip
        gt1 = gt[:, 0] # b c t h w
        
        if not is_train:
            frames_num = self.cfg["test_dataset"]["frames"]
            input_frames, out_frames = frames_num["input"], frames_num["output"]
            if input_frames != "none":
                input_frames = [int(x) for x in input_frames.split("_")]
                data["inp"] = data["inp"][:, :, input_frames]
            if out_frames != "none":
                out_frames = [int(x) for x in out_frames.split("_")]
                gt1 = gt1[:, :, out_frames]
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
                params1, params2 = output["params1"], output["params2"]
        else:
            hyponet = output

        start = time.time()
        coord = make_coord_grid(gt1.shape[2:3], (-1, 1), device=gt1.device)
        coord = einops.repeat(coord, "t d -> b t d", b=B)
        
        # Get prediction only for first clip
        pred = hyponet(coord)  # t b 3 h w
        torch.cuda.synchronize()
        dec_fps = B / (time.time() - start)

        gt1 = einops.rearrange(gt1, "b c t h w -> t b c h w")
        mses = ((pred - gt1) ** 2).view(B, -1).mean(dim=-1)
        recon_loss = mses.mean()
        loss = recon_loss
        
        # Calculate metrics for first clip only
        psnr = (-10 * torch.log10(mses)).mean()
        ssim_v = ssim(pred.flatten(end_dim=1), gt1.flatten(end_dim=1), data_range=1)

        return_dict = {
            "loss": loss.item(), # total loss may be updated later
            "recon_loss": recon_loss.item(),
            "psnr": psnr.item(),
            "ssim": ssim_v.item(),
            "enc_fps": enc_fps,
            "dec_fps": dec_fps,
            "hyponet_bpp": hyponet_bits / pred.shape[0] / pred.shape[-1] / pred.shape[-2],
            "quant_overhead_bpp": quant_overhead_bits / pred.shape[0] / pred.shape[-1] / pred.shape[-2],
        }
        
        # Add layer-wise parameter regularization loss based on mode
        param_reg_lambda_l1 = self.cfg.get("param_reg_lambda_l1")
        param_reg_lambda_l2 = self.cfg.get("param_reg_lambda_l2")
        param_reg_mode = self.cfg.get("param_reg_mode")
        param_l1_loss = 0
        param_l2_loss = 0
        
        l1_loss = nn.L1Loss()
        mse_loss = nn.MSELoss()
        
        if param_reg_lambda_l1 > 0 or param_reg_lambda_l2 > 0:
            # Compute loss between corresponding params, for each layer of hyponet  
            if param_reg_mode == 'pre_mod':
                pre_mod1, pre_mod2 = output['x_dict1'], output['x_dict2']
                for name in pre_mod1.keys():
                    if pre_mod1[name] is not None and name != 'embed':
                        param_l1_loss += l1_loss(pre_mod1[name], pre_mod2[name])
                        param_l2_loss += mse_loss(pre_mod1[name], pre_mod2[name])
            elif param_reg_mode == 'mod':
                for name in params1.keys():
                    if params1[name] is not None and name != 'embed':
                        param_l1_loss += l1_loss(params1[name], params2[name])
                        param_l2_loss += mse_loss(params1[name], params2[name])
            elif param_reg_mode == 'both':
                # Apply loss on both pre_mod and modulated params
                pre_mod1, pre_mod2 = output['x_dict1'], output['x_dict2']
                for name in pre_mod1.keys():
                    if pre_mod1[name] is not None and name != 'embed':
                        param_l1_loss += l1_loss(pre_mod1[name], pre_mod2[name])
                        param_l2_loss += mse_loss(pre_mod1[name], pre_mod2[name])
                    if params1[name] is not None and name != 'embed':
                        param_l1_loss += l1_loss(params1[name], params2[name])
                        param_l2_loss += mse_loss(params1[name], params2[name])
                param_l1_loss = 0.5 * param_l1_loss
                param_l2_loss = 0.5 * param_l2_loss
            else:
                raise ValueError(f"Invalid param_reg_mode: {param_reg_mode}")
        
            # Combine the losses
            loss += param_reg_lambda_l1 * param_l1_loss + param_reg_lambda_l2 * param_l2_loss
            return_dict['param_l1_loss'] = param_l1_loss.item()
            return_dict['param_l2_loss'] = param_l2_loss.item()

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
            specific_params = {k: v[first_elem_id] for k, v in hyponet.params.items() if v != None}
            
            specific_params_bits = utils.params_size_in_bits(specific_params)

            bpp = utils.compute_bpp(
                base_params_size_in_bits=base_params_bits,
                specific_params_size_in_bits=specific_params_bits,
                side_length=self.cfg["test_dataset"]["args"]["tubelet_size"], # not crop_size
                frame_num=self.cfg["test_dataset"]["args"]["frame_num"],
                pred_level="video",
            )
            return_dict["bpp"] = bpp
            
            # Compute bpp without base_params for comparison
            bpp_no_base = utils.compute_bpp(
                base_params_size_in_bits=0,
                specific_params_size_in_bits=specific_params_bits,
                side_length=self.cfg["test_dataset"]["args"]["tubelet_size"], # not crop_size
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
                coord = make_coord_grid(gt.shape[1:2], [-1, 1], device=gt.device)
                pred = hyponet(coord.unsqueeze(0))[:, 0]
                pred = pred.clamp(*pred_value_range)
            res.append(gt.permute(1, 0, 2, 3))
            res.append(pred)
            if self.cfg["dump_video"] not in ["no"]:
                out_vid = os.path.join(out_dir, f"{batch_id:02d}_pred_{tag}.mp4")
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
                out_path = os.path.join(out_dir, f"{batch_id:02d}_{statues}_{tag}.png")
                save_image(res[i], out_path)

    def visualize_epoch(self):
        if hasattr(self, "vislist_train"):
            self._gen_vis_result("vis_train_dataset", self.vislist_train)
        if hasattr(self, "vislist_test"):
            self._gen_vis_result("vis_test_dataset", self.vislist_test)
