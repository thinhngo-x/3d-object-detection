import torch
from pathlib import Path
import yaml
from functools import partial
import MinkowskiEngine as ME
from minke.models.backbones_3d import *
from minke.models.detection_heads import *
from minke.pipelines import *
from minke.datasets import *


if __name__ == '__main__':
    # cfg_file = Path("output/chaufferie_vanne_a_09/chaufferie.yaml")
    cfg_file = Path("output/best_epoch_160/sun_rgbd_fpn.yaml")
    with open(Path(cfg_file), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    data_cfg = cfg['Dataset']

    data_cfg['Demo_cfg']['root_dir'] = Path(data_cfg['Demo_cfg']['root_dir'])
    transforms = DataTransforms(data_cfg['Transforms'])
    data = eval(data_cfg['Name'])(**data_cfg['Demo_cfg'], transforms=transforms)
    data_cfg['Loader_cfg']['collate_fn'] = eval(data_cfg['Loader_cfg']['collate_fn'])
    dataloader, _ = prepare_data(data, False, **data_cfg['Loader_cfg'])

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
        False,
        'cuda:0'
    )
    # voxelization = Voxelization(pipeline_cfg['head_cfg']['pc_range'], pipeline_cfg['head_cfg']['voxel_size'])

    ckpt_path = Path("output/best_epoch_160/ckpt/checkpoint_155.pth")
    # ckpt_path = Path("output/chaufferie_vanne_a_09/ckpt/checkpoint_40.pth")
    pipeline.load_ckpt(ckpt_path)

    for d_batch in dataloader:
        print(d_batch['frame_id'])
        io_minke = {}
        print(d_batch['coordinates'])
        io_minke['coords'] = d_batch['coordinates']

        inp = pipeline.generate_voxels(d_batch['coordinates'], d_batch['features'])
        io_minke["preds"] = pipeline.predicts(inp)
        # pipeline.get_loss(inp, d_batch['labels'])
        io_minke['gt'] = d_batch['labels']
        torch.save(io_minke, 'io_minke.pt', _use_new_zipfile_serialization=False)
