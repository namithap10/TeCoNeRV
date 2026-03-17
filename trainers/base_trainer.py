# """
#     A basic trainer.

#     The general procedure in run() is:
#         make_datasets()
#             create . train_loader, test_loader, dist_samplers
#         make_model()
#             create . model_ddp, model
#         train()
#             create . optimizer, epoch, log_buffer
#             for epoch = 1 ... max_epoch:
#                 adjust_learning_rate()
#                 train_epoch()
#                     train_step()
#                 evaluate_epoch()
#                     evaluate_step()
#                 visualize_epoch()
#                 save_checkpoint()
# """

import os
import os.path as osp
import time

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import wandb
import yaml
from pandas import DataFrame
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import datasets
import models
import utils
from trainers import register


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


def _sanitize_cfg_for_yaml(obj):
    if isinstance(obj, dict):
        return {k: _sanitize_cfg_for_yaml(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_cfg_for_yaml(v) for v in obj]
    return obj


@register('base_trainer')
class BaseTrainer():

    def __init__(self, rank, cfg):
        self.rank = rank
        self.cfg = cfg
        self.is_master = (rank == 0)

        env = cfg['env']
        self.tot_gpus = env['world_size']
        self.distributed = env['distributed']

        # Setup log and wandb. tensorboard is intentionally disabled.
        if self.is_master:
            logger, writer, self.time_str = utils.set_save_dir(
                env['save_dir'], replace=False)
            with open(osp.join(env['save_dir'], 'cfg.yaml'), 'w') as f:
                yaml.dump(_sanitize_cfg_for_yaml(cfg), f,
                          sort_keys=False, Dumper=NoAliasDumper)

            self.log = logger.info
            self.train_psnr, self.val_psnr, self.val_ssim = [], {}, {}

            self.enable_tb = False
            self.writer = writer

            if env['wandb_upload']:
                self.enable_wandb = True
                with open('wandb.yaml', 'r') as f:
                    wandb_cfg = yaml.load(f, Loader=yaml.FullLoader)
                os.environ['WANDB_DIR'] = env['save_dir']
                wandb_kwargs = {}
                if env.get('wandb_exp_name'):
                    wandb_kwargs['name'] = env['wandb_exp_name']

                # handling evaluation mode
                if env['wandb_run_id'] != 'none':
                    self.log(f'Resuming wandb run {env["wandb_run_id"]}')
                    wandb.init(
                        id=env['wandb_run_id'],
                        resume='must',  # Ensure to resume existing run
                        project=wandb_cfg['project'],
                        entity=wandb_cfg['entity'],
                    )  # do not overwrite config or group
                else:
                    wandb.init(
                        project=wandb_cfg['project'],
                        entity=wandb_cfg['entity'],
                        group=env['exp_name'],  # exp_name = name+tag
                        config=cfg,
                        **wandb_kwargs
                    )
            else:
                self.enable_wandb = False
        else:
            self.log = lambda *args, **kwargs: None
            self.enable_tb = False
            self.enable_wandb = False

        if torch.cuda.is_available():
            torch.cuda.set_device(self.rank)
            self.device = torch.device('cuda', torch.cuda.current_device())
        else:
            self.device = torch.device('cpu')

        cudnn.benchmark = env['cudnn']

        self.info = {}
        self.log(f'Environment setup done.')
        # Log cfg
        self.log(cfg)

    def run(self):
        self.make_datasets()
        self.log('Reached end of make_datasets')
        self.starting_epoch = 1
        if self.cfg.get('eval_model') is not None:
            self.log(f"Evaluating model: {self.cfg['eval_model']}")
            model_spec = torch.load(self.cfg['eval_model'])['model']
            if not self.cfg.get('eval_same_model', True):
                self.log(
                    f"eval_same_model: {self.cfg['eval_same_model']}. Using {self.cfg['eval_saver']} as the eval saver")
                assert self.cfg['eval_saver'] is not None, "eval_saver must be specified when eval_same_model is false"
                # modify model_spec to use the eval_saver as model
                model_spec['name'] = self.cfg['eval_saver']

            self.make_model(model_spec, load_sd=True)
            self.epoch = 0
            self.log_buffer = []
            self.t_data, self.t_model = 0, 0

            if self.cfg.get('eval_residuals', False):
                self.evaluate_residuals_epoch()

            if self.cfg.get('save_weights_no_quant', False):
                residuals_type = self.cfg.get('residuals_type')
                save_path = self.cfg.get('save_path')
                target_vid = self.cfg.get('target_vid', 'jockey')
                self.save_weights_no_quant(residual_type=residuals_type, save_path=save_path, target_vid=target_vid)
        
            if self.cfg.get('save_weights_quant', False):
                residuals_type = self.cfg.get('residuals_type')
                save_path = self.cfg.get('save_path')
                target_vid = self.cfg.get('target_vid', 'jockey')
                self.save_weights_quant(residual_type=residuals_type, save_path=save_path, target_vid=target_vid)

        elif self.cfg.get('finetune_model') is not None:
            finetune_path = self.cfg.get('finetune_model')
            if os.path.exists(finetune_path):
                if self.cfg.get('finetune_same_model', True):
                    latest_ckt = torch.load(finetune_path)
                    if latest_ckt['model']['name'] != self.cfg['model']['name']:
                        latest_ckt['model']['name'] = self.cfg['model']['name']
                        latest_ckt['model']['args'] = self.cfg['model']['args']
                    self.make_model(latest_ckt['model'], load_sd=True)
                else:
                    # create model for base hypernerv, then soft load state_dict
                    # of base_ckt using finetune_spec
                    base_ckt = torch.load(finetune_path)
                    finetune_spec = {
                        'finetune_same_model': False,
                        'sd': base_ckt['model']['sd']
                    }
                    self.make_model(
                        self.cfg['model'], load_sd=False, finetune_spec=finetune_spec)

                self.starting_epoch = 1
            else:
                assert False
            self.train()
        else:
            resume_ckt_path = os.path.join(
                self.cfg['env']['save_dir'], 'epoch-last.pth')
            if 'ckt' in self.cfg:
                resume_ckt_path = self.cfg['ckt']
            if os.path.exists(resume_ckt_path):
                latest_ckt = torch.load(resume_ckt_path)
                self.make_model(latest_ckt['model'], load_sd=True)
                self.starting_epoch = latest_ckt['epoch'] + 1
            else:
                self.make_model()
            self.train()
        if self.enable_wandb:
            wandb.finish()

    def make_datasets(self):
        """
            By default, train dataset performs shuffle and drop_last.
            Distributed sampler will extend the dataset with a prefix to make the 
            length divisible by tot_gpus. Samplers should be stored in .dist_samplers.

            cfg example:

            train_dataset/test_dataset:
                name:
                args:
                loader: {batch_size: , num_workers: }
        """
        cfg = self.cfg
        self.dist_samplers = []

        def make_distributed_loader(dataset, batch_size, num_workers, shuffle=False, drop_last=False):
            sampler = DistributedSampler(
                dataset, shuffle=shuffle) if self.distributed else None
            loader = DataLoader(
                dataset,
                batch_size // self.tot_gpus,
                drop_last=drop_last,
                sampler=sampler,
                shuffle=(shuffle and (sampler is None)),
                num_workers=num_workers // self.tot_gpus,
                pin_memory=True)
            return loader, sampler

        if cfg.get('train_dataset') is not None:
            train_dataset = datasets.make(cfg['train_dataset'])
            self.log(f'Train dataset: len={len(train_dataset)}')
            self.cfg.update({'TrainSize': len(train_dataset)})
            l = cfg['train_dataset']['loader']
            self.train_loader, train_sampler = make_distributed_loader(
                train_dataset, l['batch_size'], l['num_workers'], shuffle=True, drop_last=True)
            self.dist_samplers.append(train_sampler)

        if cfg.get('test_dataset') is not None:
            l = cfg['test_dataset']['loader']
            self.test_loader_dict = {}
            if 'csv_paths' in cfg['test_dataset']:
                for dataset_name, dataset_csv in cfg['test_dataset']['csv_paths'].items():
                    test_dataset = datasets.make(cfg['test_dataset'], args={
                                                 'csv_file': dataset_csv})
                    self.log(
                        f'Test dataset: {dataset_name}, len={len(test_dataset)}')
                    self.cfg.update(
                        {f'TestSize_{dataset_name}': len(test_dataset)})
                    test_loader, test_sampler = make_distributed_loader(
                        test_dataset, l['batch_size'], l['num_workers'], shuffle=False, drop_last=False)
                    self.test_loader_dict.update({dataset_name: test_loader})
                    self.dist_samplers.append(test_sampler)
            else:
                test_dataset = datasets.make(cfg['test_dataset'])
                dataset_name = 'default'
                self.log(
                    f'Test dataset: {dataset_name}, len={len(test_dataset)}')
                self.cfg.update(
                    {f'TestSize_{dataset_name}': len(test_dataset)})
                test_loader, test_sampler = make_distributed_loader(
                    test_dataset, l['batch_size'], l['num_workers'], shuffle=False, drop_last=False)
                self.test_loader_dict.update({dataset_name: test_loader})
                self.dist_samplers.append(test_sampler)

    def make_model(self, model_spec=None, load_sd=False, finetune_spec=None):
        if model_spec is None:
            model_spec = self.cfg['model']
        model = models.make(model_spec, load_sd=load_sd)

        if finetune_spec is not None:
            assert finetune_spec['finetune_same_model'] == False, "finetune_same_model must be False for finetuning a different NeRVEnc backbone"
            missing_keys, unexpected_keys = model.load_state_dict(
                finetune_spec['sd'], strict=False)
            self.log(
                f'Loaded base finetune model successfully, missing_keys: {missing_keys}, unexpected_keys: {unexpected_keys}')

        hypernet_size = utils.compute_num_params(model, text=False)
        basemodel_size = utils.compute_num_params(model, False, False)
        embed_size = model.embed_size
        total_size = basemodel_size + embed_size
        hypersize_str = utils.text2str(hypernet_size)
        size_str = f'{utils.text2str(total_size)}_{utils.text2str(basemodel_size)}_{utils.text2str(embed_size)}'
        self.log(f'Model: #params={hypersize_str}')
        self.log(f'BaseModel: #params={size_str}')
        self.cfg.update({'HyperSize': hypersize_str, 'Size': size_str})

        if self.distributed:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model.cuda()
            model_ddp = DistributedDataParallel(model, device_ids=[self.rank])
        else:
            model.cuda()
            model_ddp = model
        self.model = model
        self.model_ddp = model_ddp

    def dump_csv(self, cfg):
        if self.is_master:
            def dump_cfg(cfg, kv_dict={}, prefix=''):
                for k, v in cfg.items():
                    if k != 'sd':
                        if isinstance(v, dict):
                            kv_dict = dump_cfg(v, kv_dict, f'{prefix}{k}_')
                        else:
                            kv_dict.update({f'{prefix}{k}': v})
                return kv_dict
            csv_dict = {}
            csv_dict = dump_cfg(cfg, csv_dict)

            def psnr_str(psnr_list, precision=2):
                return '_'.join([str(round(x, precision)) for x in psnr_list])

            def best_psnr(psnr_list, precision=2):
                return round(max(psnr_list), precision) if len(psnr_list) else 0

            csv_dict.update({'train_psnr_list': psnr_str(self.train_psnr),
                             'train_psnr': best_psnr(self.train_psnr)})
            for v_dict, v_name in zip([self.val_psnr, self.val_ssim], ['psnr', 'ssim']):
                for dataset_name, val_v in v_dict.items():
                    v_str = psnr_str(val_v, 2 if 'psnr' in v_name else 4)
                    best_v_str = best_psnr(val_v, 2 if 'psnr' in v_name else 4)
                    csv_dict.update({f'{dataset_name}_val_{v_name}_list': v_str,
                                     f'{dataset_name}_val_{v_name}': best_v_str})

            csv_path = os.path.join(
                cfg['env']['save_dir'], f'results_{self.time_str}.csv')
            csv_df = DataFrame.from_dict(csv_dict, orient='index').T
            csv_df.to_csv(csv_path)

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

    def adjust_learning_rate(self):
        base_lr = self.cfg['optimizer']['args']['lr']
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = base_lr
        self.log_temp_scalar('lr', self.optimizer.param_groups[0]['lr'])

    def log_temp_scalar(self, k, v, t=None):
        if t is None:
            t = self.epoch
        if self.enable_wandb:
            wandb.log({k: v}, step=t)

    def dist_all_reduce_mean_(self, x):
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x.div_(self.tot_gpus)

    def sync_ave_scalars_(self, ave_scalars):
        for k in ave_scalars.keys():
            x = torch.tensor(ave_scalars[k].item(
            ), dtype=torch.float32, device=self.device)
            self.dist_all_reduce_mean_(x)
            ave_scalars[k].v = x.item()
            ave_scalars[k].n *= self.tot_gpus

    def train_step(self, data):
        data = {k: v.cuda() for k, v in data.items()}
        loss = self.model_ddp(data)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}

    def train_epoch(self):
        self.model_ddp.train()
        ave_scalars = dict()

        pbar = self.train_loader
        if self.is_master:
            pbar = tqdm(pbar, desc='train', leave=True)

        t1 = time.time()
        for data in pbar:
            t0 = time.time()
            self.t_data += t0 - t1
            ret = self.train_step(data)
            self.t_model += time.time() - t0

            B = len(next(iter(data.values())))
            for k, v in ret.items():
                if ave_scalars.get(k) is None:
                    ave_scalars[k] = utils.Averager()
                ave_scalars[k].add(v, n=B)

            if self.is_master:
                pbar.set_description(
                    desc=f'train: psnr={ave_scalars["psnr"].v:.2f}')
            t1 = time.time()

        if self.distributed:
            self.sync_ave_scalars_(ave_scalars)

        logtext = 'train:'
        for k, v in ave_scalars.items():
            logtext += f' {k}={v.item():.4f}'
            self.log_temp_scalar('train/' + k, v.item())
        self.log_buffer.append(logtext)
        if self.is_master:
            self.train_psnr.append(ave_scalars['psnr'].v)

    def evaluate_step(self, data):
        data = {k: v.cuda() for k, v in data.items()}
        with torch.no_grad():
            loss = self.model_ddp(data)
        return {'loss': loss.item()}

    def evaluate_epoch(self):
        self.model_ddp.eval()
        ave_scalars = dict()

        csv_dict = {}
        base_eval_metrics = []
        for dataset_name, test_loader in self.test_loader_dict.items():
            pbar = test_loader
            if self.is_master:
                pbar = tqdm(pbar, desc=f'eval {dataset_name}', leave=True)

            t1 = time.time()
            for data in pbar:
                t0 = time.time()
                self.t_data += t0 - t1
                ret = self.evaluate_step(data)
                self.t_model += time.time() - t0

                B = len(next(iter(data.values())))
                for k, v in ret.items():
                    if ave_scalars.get(k) is None:
                        ave_scalars[k] = utils.Averager()
                    ave_scalars[k].add(v, n=B)

                if self.is_master:
                    if 'fps' in ave_scalars:
                        pbar.set_description(
                            desc=f'Eval: FPS={ave_scalars["fps"].v:.2f}, psnr={ave_scalars["psnr"].v:.2f}, ssim={ave_scalars["ssim"].v:.4f}')
                    else:
                        pbar.set_description(
                            desc=f'Eval: Enc FPS={ave_scalars["enc_fps"].v:.2f}, Dec FPS={ave_scalars["dec_fps"].v:.2f}, psnr={ave_scalars["psnr"].v:.2f}, ssim={ave_scalars["ssim"].v:.4f}')

                t1 = time.time()

            if self.distributed:
                self.sync_ave_scalars_(ave_scalars)

            logtext = '\n eval:'
            for k, v in ave_scalars.items():
                logtext += f' {dataset_name}_{k}={v.item():.4f}'

                if self.cfg.get('eval_model') is not None:
                    if self.enable_wandb:
                        base_eval_metrics.append({
                            "dataset": dataset_name,
                            "residual_type": "base_eval",
                            "metric_name": k,
                            "value": v.item()
                        })
                    csv_dict.update(
                        {f'{dataset_name}_val_{k}': f'{v.item():.4f}'})
                else:
                    # Log to wandb if not in eval_model/eval_residuals mode
                    self.log_temp_scalar(f'test/{dataset_name}_' + k, v.item())

            self.log_buffer.append(logtext)
            if self.is_master:
                if dataset_name not in self.val_psnr:
                    self.val_psnr[dataset_name] = []
                self.val_psnr[dataset_name].append(ave_scalars['psnr'].v)

        if self.is_master and self.cfg.get('eval_model') is not None:
            if not self.cfg.get('eval_residuals') and not self.cfg.get('eval_metrics_path'):
                csv_results_path = os.path.join(
                    self.cfg['env']['save_dir'], f'eval_metrics.csv')
            else:
                csv_results_path = os.path.join(
                    self.cfg['eval_metrics_path'], f'eval_metrics.csv')
            csv_df = DataFrame.from_dict(csv_dict, orient='index').T
            csv_df.to_csv(csv_results_path)
            self.log(f'Eval metrics saved to {csv_results_path}')
            self.log(', '.join(self.log_buffer))

    def evaluate_residuals_epoch(self):
        if self.is_master and self.cfg.get('eval_model') is not None:
            # Handle possible list of quant_bits e.g. '8_6_4' -> [8, 6, 4]
            quant_bit_config = self.cfg.get('quant_bit', 8)
            if isinstance(quant_bit_config, str):
                quant_bits_list = [int(bit)
                                   for bit in quant_bit_config.split('_')]
            elif isinstance(quant_bit_config, int):
                quant_bits_list = [quant_bit_config]

            self.log(f"Evaluating residuals for quant_bits: {quant_bits_list}")

            csv_path = None
            encoding_type = self.cfg.get('encoding_type', 'arithmetic')
            quant_axis = int(self.cfg.get('quant_axis', 0))
            eval_csv_prefix = self.cfg.get('eval_csv_prefix', '')
            log_per_video = self.cfg.get('eval_log_per_video', False)
            chunk_pred_batch_size = self.cfg.get('chunk_pred_batch_size', None)

            for qbit in quant_bits_list:
                _, csv_path = self.evaluate_x_dict_residuals_epoch(
                    quant_bit=qbit,
                    encoding_type=encoding_type,
                    quant_axis=quant_axis,
                    eval_csv_prefix=eval_csv_prefix,
                    log_per_video=log_per_video,
                    chunk_pred_batch_size=chunk_pred_batch_size,
                )
            self.log(f'Residuals evaluation saved to {csv_path}')

    def save_checkpoint(self, filename):
        if not self.is_master:
            return
        model_spec = self.cfg['model']
        model_spec['sd'] = self.model.state_dict()
        optimizer_spec = self.cfg['optimizer']
        optimizer_spec['sd'] = self.optimizer.state_dict()
        checkpoint = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': self.epoch,
            'cfg': self.cfg,
        }
        torch.save(checkpoint, osp.join(self.cfg['env']['save_dir'], filename))
