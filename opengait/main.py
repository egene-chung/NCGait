
import os
import argparse
import torch
import torch.nn as nn
from modeling import models
from utils import config_loader, get_ddp_module, init_seeds, params_count, get_msg_mgr

parser = argparse.ArgumentParser(description='Main program for opengait.')
parser.add_argument('--local_rank', type=int, default=0,
                    help="passed by torch.distributed.launch module")
parser.add_argument('--local-rank', type=int, default=0,
                    help="passed by torch.distributed.launch module, for pytorch >=2.0")
parser.add_argument('--cfgs', type=str,
                    default='configs/ccpg/deepgaitv2_ccpg-ncb-pgfc-vm.yaml', help="path of config file")
parser.add_argument('--phase', default='train',
                    choices=['train', 'test'], help="choose train or test phase")
parser.add_argument('--log_to_file', action='store_true',
                    help="log to file, default path is: <output_root>/<dataset>/<model>/<save_name>/<logs>/<Datetime>.txt")
parser.add_argument('--iter', default=0, help="iter to restore")
parser.add_argument('--seed', type=int, default=None,
                    help="base random seed; a non-zero seed also appends -seed<N> to save_name")
opt = parser.parse_args()


def initialization(cfgs, training):
    msg_mgr = get_msg_mgr()
    engine_cfg = cfgs['trainer_cfg'] if training else cfgs['evaluator_cfg']
    output_root = cfgs.get('env_cfg', {}).get('output_root') or './output'
    output_path = os.path.join(output_root, cfgs['data_cfg']['dataset_name'],
                               cfgs['model_cfg']['model'], engine_cfg['save_name'])
    if training:
        msg_mgr.init_manager(output_path, opt.log_to_file, engine_cfg['log_iter'],
                             engine_cfg['restore_hint'] if isinstance(engine_cfg['restore_hint'], (int)) else 0)
    else:
        msg_mgr.init_logger(output_path, opt.log_to_file)

    msg_mgr.log_info(
        "Run %s | dataset=%s | model=%s | save_name=%s" % (
            "train" if training else "test",
            cfgs['data_cfg']['dataset_name'],
            cfgs['model_cfg']['model'],
            engine_cfg['save_name']))

    # Base seed 0 keeps the original per-rank seeds (0, 1, ...); seed N
    # shifts every rank so no two (seed, rank) pairs collide.
    seed = torch.distributed.get_rank() + \
        engine_cfg.get('seed', 0) * torch.distributed.get_world_size()
    # Enable CUDA deterministic for perfect reproducibility
    cuda_deterministic = engine_cfg.get('cuda_deterministic', True)
    init_seeds(seed, cuda_deterministic=cuda_deterministic)


def run_model(cfgs, training):
    msg_mgr = get_msg_mgr()
    model_cfg = cfgs['model_cfg']
    Model = getattr(models, model_cfg['model'])
    model = Model(cfgs, training)
    if training and cfgs['trainer_cfg']['sync_BN']:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if cfgs['trainer_cfg']['fix_BN']:
        model.fix_BN()
    model = get_ddp_module(model, cfgs['trainer_cfg']['find_unused_parameters'])
    msg_mgr.log_info(params_count(model))
    msg_mgr.log_info("Model ready.")

    if training:
        Model.run_train(model)
    else:
        Model.run_test(model)


if __name__ == '__main__':
    torch.distributed.init_process_group('nccl', init_method='env://')
    if torch.distributed.get_world_size() != torch.cuda.device_count():
        raise ValueError("Expect number of available GPUs({}) equals to the world size({}).".format(
            torch.cuda.device_count(), torch.distributed.get_world_size()))
    cfgs = config_loader(opt.cfgs)
    if opt.iter != 0:
        cfgs['evaluator_cfg']['restore_hint'] = int(opt.iter)
        cfgs['trainer_cfg']['restore_hint'] = int(opt.iter)
    if opt.seed is not None:
        for engine in ('trainer_cfg', 'evaluator_cfg'):
            cfgs[engine]['seed'] = opt.seed
            if opt.seed != 0:
                cfgs[engine]['save_name'] += '-seed%d' % opt.seed

    training = (opt.phase == 'train')
    initialization(cfgs, training)
    run_model(cfgs, training)
