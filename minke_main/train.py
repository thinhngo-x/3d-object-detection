from minke.datasets.transforms import DataTransforms
import torch
from minke.pipelines import *
from minke.datasets import *
import torch
import yaml
from pathlib import Path
from functools import partial
from tqdm import tqdm
import argparse
import os
from torch.utils.tensorboard import SummaryWriter
from benchmark import eval_single_ckpt
from minke.utils import common_utils
import shutil
import json
CUR_DIR = Path(os.getcwd())


class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg_file', help="Path to config file")
    parser.add_argument('--tag', help="Tag of experiment", default='default', type=str)
    parser.add_argument('--epoch', help="Number of epochs", default=320, type=int)
    parser.add_argument('--launcher', choices=['none', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distributed training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    # torch.autograd.set_detect_anomaly(True)
    with open(Path(args.cfg_file), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
        LOCAL_RANK = 0
    else:
        total_gpus, LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True
        torch.manual_seed(0)

    out_dir = CUR_DIR / 'output' / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    if not (out_dir / Path(args.cfg_file).name).exists():
        shutil.copy(Path(args.cfg_file), out_dir)

    data_cfg = cfg['Dataset']

    batch_size = data_cfg['Loader_cfg']['batch_size']
    assert batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
    batch_size = batch_size // total_gpus

    data_cfg['Cfg']['root_dir'] = Path(data_cfg['Cfg']['root_dir'])
    transforms = DataTransforms(data_cfg['Transforms'])
    data = eval(data_cfg['Name'])(**data_cfg['Cfg'], transforms=transforms)
    data_cfg['Loader_cfg']['collate_fn'] = eval(data_cfg['Loader_cfg']['collate_fn'])
    dataloader, sampler = prepare_data(data, dist_train, **data_cfg['Loader_cfg'])

    data_cfg['Test_cfg']['root_dir'] = Path(data_cfg['Test_cfg']['root_dir'])
    transforms = DataTransforms(data_cfg['Transforms'])
    data = eval(data_cfg['Name'])(**data_cfg['Test_cfg'], transforms=transforms)
    data_cfg['Loader_cfg']['batch_size'] = 16
    test_dataloader, _ = prepare_data(data, dist_train, **data_cfg['Loader_cfg'])

    # quant_mode = ME.SparseTensorQuantizationMode[data_cfg['Quantization']['mode']]
    # voxel_size = data_cfg['Quantization']['voxel_size']
    # max_pts_per_vox = data_cfg['Quantization']['max_points_per_vox']

    pipeline_cfg = cfg['Pipeline']
    optim = partial(eval(pipeline_cfg['optimizer']['name']), **pipeline_cfg['optimizer']['optim_cfg'])
    pipeline = eval(pipeline_cfg['Name'])(
        pipeline_cfg['backbone_cfg'],
        pipeline_cfg['head_cfg'],
        pipeline_cfg['voxel_cfg'],
        optim,
        True,
        'cuda',
        steps_per_epoch=len(dataloader),
        epochs=args.epoch
    )
    if dist_train:
        pipeline = MyDataParallel(
            pipeline, device_ids=[LOCAL_RANK % torch.cuda.device_count()]
        )
    # grid_size = pipeline_cfg['head_cfg']['grid_size']
    writer = SummaryWriter(log_dir=out_dir / 'tensorboard') if LOCAL_RANK == 0 else None
    init_epoch = -1
    if (out_dir / 'ckpt').exists() and len(os.listdir(out_dir / 'ckpt')) > 0:
        ckpt_saved = os.listdir(out_dir / 'ckpt')
        ckpt_saved = [int(e.split('.')[0].split('_')[1]) for e in ckpt_saved]
        latest_ckpt = out_dir / 'ckpt' / ('checkpoint_' + str(max(ckpt_saved)) + '.pth')
        init_epoch = pipeline.load_ckpt(latest_ckpt)

    eval_cfg = cfg['Eval']

    iter = pipeline.cur_iter
    # alpha = .14
    # alpha_incre = .01
    for i in tqdm(range(init_epoch + 1, args.epoch)):
        smooth_loss_cls = 0.
        smooth_loss_reg = 0.
        smooth_loss_seg = 0.
        if dist_train:
            sampler.set_epoch(i)
        for d_batch in dataloader:
            torch.cuda.empty_cache()
            # print(d_batch['frame_id'])
            iter += 1
            # print(torch.unique(d_batch['coordinates'][:, 0]))
            inp = pipeline.generate_voxels(d_batch['coordinates'], d_batch['features'])
            loss_cls, loss_reg, loss_objness = pipeline.get_loss(inp, d_batch['labels'], writer=writer)
            loss = loss_cls + sum(loss_reg) + loss_objness

            pipeline.optimizer.zero_grad()
            loss.backward()
            grad_norm, lr = pipeline.step()
            if writer is not None:
                writer.add_scalars('train/loss', {'loss_cls': loss_cls.item(),
                                                  'loss_reg': loss_reg[0].item(),
                                                  'loss_dir': loss_reg[1].item(),
                                                  'loss_corners': loss_reg[2].item(),
                                                  'loss_seg': loss_objness.item(),
                                                  'loss_total': loss.item()},
                                   global_step=iter)
                writer.add_scalar('train/grad_norm', grad_norm, global_step=iter)
                if lr is not None:
                    writer.add_scalar('train/lr', lr, global_step=iter)
            # print('CLS ', loss_cls)
            # print('REG ', sum(loss_reg))
            # print('OBJ ', loss_objness)
        ckpt = pipeline.save_ckpt(out_dir / 'ckpt', i, interval=eval_cfg['save_interval'])
        if ckpt is not None:
            torch.cuda.empty_cache()
            pipeline.eval()
            with torch.no_grad():
                ret_dict = eval_single_ckpt(pipeline, test_dataloader, ckpt, eval_cfg['iou_thres'], data_cfg['Test_cfg']['class_names'])
            if writer is not None:
                writer.add_scalar('eval/ap', ret_dict['map_11_point'], global_step=iter)
            p = out_dir / 'eval'
            p.mkdir(parents=True, exist_ok=True)
            if LOCAL_RANK == 0:
                with open(p / ('checkpoint_' + str(i) + '.txt'), 'w') as f:
                    f.write(json.dumps(ret_dict))
            pipeline.train()
            torch.cuda.empty_cache()
    writer.close()
