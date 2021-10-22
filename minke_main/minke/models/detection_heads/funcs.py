def gen_anchor_sizes(anchor_cfg):
    """Generate a list of possible anchors per voxel.

    Args:
        anchor_cfg (dict): [description]
            {
                class_name:
                    {
                        'ratio_xy': List[float],
                        'size_xy': List[float],
                        'size_z': List[float],
                        'dir': List[float]
                    }
            }

    Returns:
        anchors_per_vox (List[List[float, 4]])
    """
    anchors_per_vox = []
    for _, cfg in anchor_cfg.items():
        sizes_xy = []
        for ratio in cfg['ratio_xy']:
            for w in cfg['size_xy']:
                sizes_xy.append([w, w * ratio])
        for xy in sizes_xy:
            for z in cfg['size_z']:
                for d in cfg['dir']:
                    anchors_per_vox.append([xy[0], xy[1], z, d])

    return anchors_per_vox
