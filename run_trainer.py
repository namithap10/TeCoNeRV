"""
    Generate a cfg object according to a cfg file and args, then spawn Trainer(rank, cfg).
"""

import argparse
import copy
import datetime
import os
import random

import numpy as np
import torch
import yaml
from mergedeep import merge

import trainers
import utils


def parse_args():
    def input_size_type(x):
        x = x.strip('"\'')
        
        if '_' in x:
            x = x.replace('_', 'x')
            
        try:
            return int(x)
        except ValueError:
            if 'x' in x:
                parts = x.split('x')
            else:
                raise argparse.ArgumentTypeError(f"input_size must be either an integer or a string with format 'height_width' or 'heightxwidth'")
            
            return [int(parts[0]), int(parts[1])]

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg')
    parser.add_argument('--data_path', default='data/k400')
    parser.add_argument('--csv_file', default='k400_train.js')
    parser.add_argument('--eval_frames', type=str, default='none')
    parser.add_argument('--frame_num', type=int, default=4)
    parser.add_argument('--input_size', type=input_size_type, default=(128, 128))
    parser.add_argument('--batch_size', '-b', type=int, default=16)
    parser.add_argument('--num_workers', '-j', type=int, default=16)
    parser.add_argument('--out_path', default='output path')
    parser.add_argument('--name', '-n', default=None)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--instance_tag', type=str, default='')
    parser.add_argument('--wandb-exp-name', type=str, default=None)
    parser.add_argument('--cudnn', action='store_true')
    parser.add_argument('--replace', action='store_true')
    parser.add_argument('--port-offset', '-p', type=int, default=0)
    parser.add_argument('--wandb-upload', '-w', action='store_true')
    parser.add_argument('--wandb-run-id', '-r', type=str, default='none')
    parser.add_argument('--opts', type=str, nargs='*', default=[], help='cfg args to update')
    parser.add_argument('--manualSeed', type=int, default=1, help='manual seed')
    parser.add_argument('--tubelet_size', type=input_size_type, default=(160, 160))
    args = parser.parse_args()

    return args

def make_cfg(args):
    with open(args.cfg, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    def translate_cfg_(d):
        for k, v in d.items():
            if isinstance(v, dict):
                translate_cfg_(v)
            elif isinstance(v, str):
                if v.startswith('$') and v.endswith('$'):
                    v = copy.deepcopy(getattr(args, v.replace('$', '')))
                d[k] = v
    translate_cfg_(cfg)

    if args.name is None:
        exp_name = os.path.basename(args.cfg).split('.')[0]
    else:
        exp_name = args.name
    exp_name += args.tag
    
    timestamp = datetime.datetime.now().strftime("%m-%d_%H-%M")

    env = dict()
    env['exp_name'] = exp_name # used as group name for wandb
    if args.instance_tag != '':
        env['save_dir'] = os.path.join(args.out_path, exp_name, args.instance_tag)
        env['instance_tag'] = args.instance_tag
    else:
        env['save_dir'] = os.path.join(args.out_path, exp_name, timestamp)
        
    print(f"save_dir: {env['save_dir']}")
    
    env['tot_gpus'] = torch.cuda.device_count()
    env['cudnn'] = args.cudnn
    if args.port_offset == 0:
        env['port'] = str(29500 + np.random.randint(0, 100))
    else:
        env['port'] = str(29500 + args.port_offset)
    env['wandb_upload'] = args.wandb_upload
    env['wandb_exp_name'] = args.wandb_exp_name
    env['wandb_run_id'] = args.wandb_run_id
    cfg['env'] = env

    def build_tree(tree_list):
        if len(tree_list)>=2:
            return {tree_list[0]: build_tree(tree_list[1:]) if len(tree_list)>2 else tree_list[-1]}

    def nested_v(dict, keys):
        for key in keys:
            dict = dict[key]
        return dict

    def convert(target_type, x):
        if x.lower() == 'none': 
            return None

        if target_type == bool:
            return x.lower() in ('true', '1', 'yes')

        if target_type == int and isinstance(x, str) and '_' in x:
            return x

        try:
            return target_type(x)
        except (ValueError, TypeError) as e:
            print(f"Warning: Could not convert '{x}' to type {target_type}. Using original string. Error: {e}")
            return x

    assert len(args.opts) % 2 == 0
    for cur_cfg_key, v in zip(args.opts[::2], args.opts[1::2]):
        keys = cur_cfg_key.split('.')
        v = convert(type(nested_v(cfg, keys)), v)
        cfg = merge(cfg, build_tree(keys+[v]))

    return cfg


def main():
    if "RANK" in os.environ:
        if int(os.environ["RANK"]) != 0:
            import sys
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')
    
    args = parse_args()
    cfg = make_cfg(args)
    init_distributed_mode(cfg)
    
    setup_for_distributed(cfg['env']['rank'] == 0)
    
    if cfg['env']['rank'] == 0:
        utils.ensure_path(cfg['env']['save_dir'], args.replace)
    if cfg['env']['distributed']:
        torch.distributed.barrier()
        
    torch.manual_seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    random.seed(args.manualSeed)
    seed_everything(args.manualSeed)
    
    # Run trainer
    main_worker(cfg)

def init_distributed_mode(cfg):
    """Initialize distributed training settings"""
    env = cfg['env']
    
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        env['rank'] = int(os.environ["RANK"])
        env['world_size'] = int(os.environ["WORLD_SIZE"]) 
        env['gpu'] = int(os.environ["LOCAL_RANK"]) if torch.cuda.is_available() else None
        # Use MASTER_PORT from environment if available
        env['port'] = os.environ.get("MASTER_PORT", env.get('port', '29500'))
    else:
        print("Not using distributed mode")
        env['distributed'] = False
        env['rank'] = 0
        env['world_size'] = 1
        env['gpu'] = None
        return

    env['distributed'] = True
    
    if torch.cuda.is_available():
        torch.cuda.set_device(env['gpu'])
        env['dist_backend'] = "nccl"
    else:
        env['dist_backend'] = "gloo"

    if env['rank'] == 0:
        print(f"| distributed init (rank {env['rank']}, world {env['world_size']}): env://", flush=True)
    
    torch.distributed.init_process_group(
        backend=env['dist_backend'],
        init_method='env://',
        world_size=env['world_size'],
        rank=env['rank'],
        timeout=datetime.timedelta(days=365)
    )
    torch.distributed.barrier()

def main_worker(cfg):
    """Main worker function that creates and runs the trainer"""
    trainer = trainers.trainers_dict[cfg['trainer']](cfg['env']['rank'], cfg)
    trainer.run()

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    main()
