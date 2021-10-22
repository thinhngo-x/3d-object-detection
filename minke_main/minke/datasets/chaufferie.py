import torch
from pathlib import Path
import numpy as np


class ChaufferieDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, data_dir, label_dir, idx_file, class_names,
                 transforms=None, mode='train'):
        """Initialize.

        Args:
            root_dir (pathlib.Path): Path to the directory of the dataset.
            data_dir (String or pathlib.Path): Name of dir containing 3D scans
            idx_file (String or pathlib.Path): Name of file containing sampled indices
            class_names (List[String]): Names of classes to be used.
            transforms (datasets.transforms, optional): Data augmentor. Defaults to None.
        """
        self.mode = mode
        self.class_names = class_names
        self.label_dir = Path(root_dir / label_dir)
        self.data_dir = Path(root_dir / data_dir)
        if mode == 'train':
            self.training = True
        else:
            self.training = False
        if mode == 'train' or mode == 'test':
            assert(Path(root_dir / idx_file).exists())
            with open(Path(root_dir / idx_file), 'r') as f:
                self.sampled_idx = [
                    x.strip() for x in f.readlines()
                ]
        elif mode == 'demo':
            self.sampled_idx = idx_file
        self.transforms = transforms

    def __len__(self):
        return len(self.sampled_idx)

    def get_pc(self, idx):
        """Load point cloud of the scene idx."""
        pc_file = self.data_dir / (idx + '.npy')
        assert pc_file.exists()
        coords = np.load(str(pc_file))
        pc = np.zeros((coords.shape[0], 4))
        pc[:, :3] = coords
        pc[:, 3] = 1.
        return pc

    def get_label(self, idx):
        """Load ground truth labels of the scene idx."""
        label_file = self.label_dir / (idx + '.txt')
        if not label_file.exists():
            return {'gt_boxes': np.array([]), 'gt_names': np.array([])}

        def read_single_line(line):
            data = line.split(' ')
            if data[0] not in self.class_names:
                return None, None
            data[1:] = [float(x) for x in data[1:]]
            gt_box = [d for d in data[1:]]
            return gt_box, data[0]

        def read_chaufferie_label(label_file):
            with open(label_file) as f:
                lines = [line.rstrip() for line in f]
            gt_boxes = []
            gt_names = []
            for line in lines:
                gt_box, gt_name = read_single_line(line)
                if gt_box is None:
                    continue
                gt_boxes.append(gt_box)
                gt_names.append(gt_name)
            gt_boxes = np.array(gt_boxes, dtype=np.float32)
            gt_names = np.array(gt_names)
            return {'gt_boxes': gt_boxes, 'gt_names': gt_names}

        return read_chaufferie_label(label_file)

    def encode_label(self, label):
        """Generate labels for computing loss based on the original labels.

        Args:
            label (Dict): Dict of label.
                {
                    'gt_boxes': np.array of gt boxes (#gt, box_size)
                    'gt_names': np.array of gt classes (#gt)
                }

        Returns:
            label (Dict): Transformed label.
                {
                    'gt_boxes': torch.Tensor of gt boxes (#gt, box_size)
                    'gt_names': np.array of gt classes (#gt)
                    'gt_cls_code': torch.Tensor of shape (#gt)
                }
        """
        gt_names = label['gt_names']
        gt_codes = torch.tensor([self.class_names.index(name) for name in gt_names], dtype=int)
        label['gt_cls_code'] = gt_codes
        label['gt_boxes'] = torch.from_numpy(label['gt_boxes'])

        return label

    def __getitem__(self, idx):
        idx = self.sampled_idx[idx]
        pc = self.get_pc(idx)
        label = self.get_label(idx)
        if label['gt_names'].shape[0] <= 0 or pc.shape[0] <= 1000:
            # if self.training:
            idx = np.random.randint(len(self.sampled_idx))
            return self.__getitem__(idx)
        if not self.training:
            pc, label = self.transforms.translate_to_origin(pc, label)

        if self.training and self.transforms is not None:
            pc, label = self.transforms.transform(pc, label)

        if self.training and (label['gt_names'].shape[0] <= 0 or pc.shape[0] <= 1000):
            idx = np.random.randint(len(self.sampled_idx))
            return self.__getitem__(idx)

        self.encode_label(label)

        item = {
            'coords': pc[:, :3],
            'features': pc[:, 3][:, None],
            'labels': label,
            'frame_id': idx
        }
        return item
