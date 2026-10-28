#!/usr/bin/env python3
"""Evaluate a trained checkpoint (IoU, Pd, Fa) on the dense test masks."""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from grsr.common import resolve_split_path
from grsr.networks import available_networks, build_network
from grsr.training.metrics import mIoU, PD_FA


class TestSet(Dataset):
    def __init__(self, img_dir, gt_dir, ids, size=256, suffix='.png'):
        self.img_dir, self.gt_dir, self.ids = img_dir, gt_dir, ids
        self.size, self.suffix = size, suffix
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        name = self.ids[i]
        img = Image.open(os.path.join(self.img_dir, name + self.suffix)).convert('RGB')
        mask = Image.open(os.path.join(self.gt_dir, name + self.suffix))
        img = img.resize((self.size, self.size), Image.BILINEAR)
        mask = mask.resize((self.size, self.size), Image.NEAREST)
        # Preserve uint8 so ToTensor converts image values from [0,255] to [0,1].
        img = self.tf(np.array(img, dtype=np.uint8))
        mask = np.expand_dims(np.array(mask, dtype=np.float32), 0) / 255.0
        return img, torch.from_numpy(mask)


def data_dirs(root):
    if os.path.isdir(os.path.join(root, 'IRSTD1k_Img')):
        return (os.path.join(root, 'IRSTD1k_Img'),
                os.path.join(root, 'IRSTD1k_Label'))
    return (os.path.join(root, 'images'), os.path.join(root, 'masks'))


def main():
    ap = argparse.ArgumentParser(description='Evaluate a downstream checkpoint on dense masks')
    ap.add_argument('--model', type=str, required=True)
    ap.add_argument('--data_root', type=str, required=True)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--gpus', type=str, default='0')
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--test_split', type=str, default='test.txt')
    ap.add_argument('--base_size', type=int, default=256)
    ap.add_argument('--network', type=str, default='dnanet',
                    choices=available_networks())
    ap.add_argument('--channel_size', type=str, default='three')
    ap.add_argument('--backbone', type=str, default='resnet_18')
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(resolve_split_path(Path(args.data_root), args.test_split)) as f:
        test_ids = [l.strip() for l in f if l.strip()]

    img_dir, gt_dir = data_dirs(args.data_root)
    model = build_network(
        name=args.network,
        channel_size=args.channel_size,
        backbone=args.backbone,
        deep_supervision=True,
    )
    try:
        state_dict = torch.load(args.model, map_location=device, weights_only=True)
    except TypeError:  # PyTorch < 2.0, including the server's torch 1.8 env
        state_dict = torch.load(args.model, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    loader = DataLoader(TestSet(img_dir, gt_dir, test_ids, size=args.base_size),
                       batch_size=args.batch_size, num_workers=args.num_workers,
                       pin_memory=device.type == 'cuda')
    metric = mIoU(1)
    pdfa = PD_FA(1, 10)

    with torch.no_grad():
        for data, labels in tqdm(loader, ncols=80):
            data, labels = data.to(device), labels.to(device)
            output = model(data)
            pred = output[-1] if isinstance(output, (list, tuple)) else output
            metric.update(pred, labels)
            for i in range(pred.shape[0]):
                # PD_FA thresholds scores in [0, 255]; model outputs are logits.
                pdfa.update(torch.sigmoid(pred[i, 0]) * 255,
                            labels[i, 0] * 255)

    _, miou = metric.get()
    FA, PD = pdfa.get(len(test_ids))
    tag = os.path.basename(os.path.dirname(os.path.abspath(args.model)))

    print(f'\n[{tag}]  mIoU: {miou:.4f}')
    print('Pd / Fa @ bins:')
    for i in range(len(PD)):
        print(f'  bin {i}: Pd={PD[i]:.4f}  Fa={FA[i]:.6e}')


if __name__ == '__main__':
    main()
