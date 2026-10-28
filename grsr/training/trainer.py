#!/usr/bin/env python3
"""GRSR data-driven iterative label refinement and downstream evaluation.

This file implements Sec. III-B and the training settings stated in Sec. IV:
continuous asymmetric loss, asymmetric SoftIoU, saliency peak attention, and
three rounds of confidence-gated pseudo-mask updates.
"""

import argparse
import datetime as dt
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageOps
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from scipy import ndimage as ndi
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from grsr.common import resolve_split_path
from grsr.networks import available_networks, build_network
from grsr.training.metrics import PD_FA, ROCMetric, mIoU


SUP_SUBDIR = ("supervision_maps_grsr", "initial")


def _as_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def apply_confidence_update(
    current,
    probability,
    gamma_high=0.9,
    gamma_low=0.2,
    update_mode="union",
    low_decay=0.25,
    protection_threshold=0.8,
    prior=None,
    anchor_beta=0.25,
    prior_core=0.5,
):
    """Apply one confidence-gated pseudo-label update as a pure array op."""
    if update_mode not in {"union", "decay", "high_only", "anchor"}:
        raise ValueError(f"Unknown update_mode: {update_mode}")
    if not 0.0 <= low_decay <= 1.0:
        raise ValueError("low_decay must be in [0, 1]")
    if gamma_low >= gamma_high:
        raise ValueError("gamma_low must be smaller than gamma_high")
    current = np.asarray(current, dtype=np.float32)
    probability = np.asarray(probability, dtype=np.float32)
    high = probability > gamma_high
    low = probability < gamma_low
    if update_mode == "anchor":
        # Prior-anchored bidirectional fusion. Confident foreground and confident
        # background both overwrite the current value, so an erroneous foreground
        # response can be removed. The physics-inspired prior G_0 keeps a fixed
        # share of every pixel, so the labels cannot drift away from it, and
        # background evidence is never allowed to erase the prior core.
        if prior is None:
            raise ValueError("anchor mode requires the initial prior G_0")
        prior = np.asarray(prior, dtype=np.float32)
        fused = current.copy()
        fused[high] = probability[high]
        pull_down = low & (prior < prior_core)
        fused[pull_down] = probability[pull_down]
        return np.clip(anchor_beta * prior + (1.0 - anchor_beta) * fused, 0.0, 1.0)
    if update_mode == "high_only":
        # Preserve the current pseudo-label and add only confident foreground
        # responses.  Ambiguous and low-confidence predictions are ignored,
        # which is the conservative ablation used to test the manuscript's
        # "ambiguous pixels are not promoted" interpretation.
        return np.maximum(current, np.where(high, probability, 0.0))
    if update_mode == "union":
        return np.maximum(current, np.where(high | low, probability, 0.0))

    updated = current.copy()
    updated[high] = np.maximum(current[high], probability[high])
    eligible = low & (current < protection_threshold)
    updated[eligible] = (
        (1.0 - low_decay) * current[eligible]
        + low_decay * probability[eligible]
    )
    return np.clip(updated, 0.0, 1.0)


class PointSupTrainSet(Dataset):
    """Training images paired with continuous GRSR pseudo masks."""

    def __init__(
        self,
        image_dir,
        mask_dir,
        image_ids,
        base_size=256,
        crop_size=256,
        transform=None,
        suffix=".png",
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_ids = image_ids
        self.base_size = base_size
        self.crop_size = crop_size
        self.transform = transform
        self.suffix = suffix

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        name = self.image_ids[index]
        image = Image.open(os.path.join(self.image_dir, name + self.suffix)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, name + self.suffix)).convert("L")

        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        width, height = image.size
        if height > width:
            out_height = long_size
            out_width = int(width * long_size / height)
        else:
            out_width = long_size
            out_height = int(height * long_size / width)
        image = image.resize((out_width, out_height), Image.BILINEAR)
        mask = mask.resize((out_width, out_height), Image.BILINEAR)

        pad_height = max(0, self.crop_size - out_height)
        pad_width = max(0, self.crop_size - out_width)
        if pad_height or pad_width:
            image = ImageOps.expand(image, border=(0, 0, pad_width, pad_height), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, pad_width, pad_height), fill=0)

        width, height = image.size
        x0 = random.randint(0, width - self.crop_size)
        y0 = random.randint(0, height - self.crop_size)
        image = image.crop((x0, y0, x0 + self.crop_size, y0 + self.crop_size))
        mask = mask.crop((x0, y0, x0 + self.crop_size, y0 + self.crop_size))

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        image = np.array(image, dtype=np.uint8)
        target = np.expand_dims(np.array(mask, dtype=np.float32), axis=0) / 255.0
        if self.transform:
            image = self.transform(image)
        return image, torch.from_numpy(target)


class PointSupTestSet(Dataset):
    def __init__(
        self,
        image_dir,
        mask_dir,
        image_ids,
        base_size=256,
        transform=None,
        suffix=".png",
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_ids = image_ids
        self.base_size = base_size
        self.transform = transform
        self.suffix = suffix

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        name = self.image_ids[index]
        image = Image.open(os.path.join(self.image_dir, name + self.suffix)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, name + self.suffix)).convert("L")
        image = image.resize((self.base_size, self.base_size), Image.BILINEAR)
        mask = mask.resize((self.base_size, self.base_size), Image.NEAREST)
        image = np.array(image, dtype=np.uint8)
        target = np.expand_dims(np.array(mask, dtype=np.float32), axis=0) / 255.0
        if self.transform:
            image = self.transform(image)
        return image, torch.from_numpy(target)


def baseline_continuous_loss(logits, target, eps=1e-6):
    """Ablation baseline from Table II: L_MSE + L_SoftIoU."""
    probability = torch.sigmoid(logits)
    mse = torch.mean((probability - target) ** 2)
    probability = probability.reshape(probability.size(0), -1)
    target = target.reshape(target.size(0), -1)
    intersection = torch.sum(probability * target, dim=1)
    union = torch.sum(probability + target - probability * target, dim=1)
    soft_iou = torch.mean(1.0 - (intersection + eps) / (union + eps))
    return mse + soft_iou


class GRSRContinuousLoss(nn.Module):
    """Equations (8)-(12) from the manuscript."""

    def __init__(
        self,
        w_fn=3.0,
        w_fp=1.0,
        focusing_gamma=2.0,
        aiou_alpha=0.3,
        temperature=0.1,
        eps=1e-6,
        aiou_band=0,
        band_threshold=0.1,
        lambda_a=1.0,
        lambda_s=0.5,
    ):
        super().__init__()
        self.lambda_a = lambda_a
        self.lambda_s = lambda_s
        if w_fn <= w_fp:
            raise ValueError("The manuscript requires w_fn > w_fp")
        self.w_fn = w_fn
        self.w_fp = w_fp
        self.focusing_gamma = focusing_gamma
        self.aiou_alpha = aiou_alpha
        self.temperature = temperature
        self.eps = eps
        self.aiou_band = int(aiou_band)
        self.band_threshold = band_threshold

    def forward(self, logits, target):
        alpha_map = None
        if self.aiou_band > 0:
            # Boundary-local tolerance: false positives are discounted only inside
            # a band of aiou_band pixels around the soft target support, as the
            # manuscript motivates; outside it they are penalised in full.
            spatial = target if target.dim() == 4 else target.unsqueeze(1)
            support = (spatial > self.band_threshold).float()
            band = nn.functional.max_pool2d(
                support, kernel_size=2 * self.aiou_band + 1, stride=1, padding=self.aiou_band
            )
            alpha_map = (self.aiou_alpha * band + (1.0 - band)).reshape(target.size(0), -1)
        probability = torch.sigmoid(logits).clamp(self.eps, 1.0 - self.eps)
        probability = probability.reshape(probability.size(0), -1)
        target = target.reshape(target.size(0), -1)

        positive = (
            -self.w_fn
            * target
            * torch.pow(1.0 - probability, self.focusing_gamma)
            * torch.log(probability)
        )
        negative = (
            -self.w_fp
            * (1.0 - target)
            * torch.pow(probability, self.focusing_gamma)
            * torch.log(1.0 - probability)
        )
        continuous_asymmetric = torch.mean(positive + negative)

        intersection = torch.sum(probability * target, dim=1)
        alpha = self.aiou_alpha if alpha_map is None else alpha_map
        asymmetric_union = torch.sum(
            target + alpha * probability * (1.0 - target), dim=1
        )
        asymmetric_soft_iou = torch.mean(
            1.0 - (intersection + self.eps) / (asymmetric_union + self.eps)
        )

        attention = torch.softmax(probability / self.temperature, dim=1)
        attended_peak = torch.sum(attention * target * probability, dim=1)
        saliency_peak_attention = torch.mean((1.0 - attended_peak) ** 2)

        # Sec. IV fixes lambda_a=1 and lambda_s=0.5.
        return continuous_asymmetric + self.lambda_a * asymmetric_soft_iou + self.lambda_s * saliency_peak_attention


class PointSupTrainer:
    def __init__(self, args):
        self.args = args
        self.args.deep_supervision = _as_bool(args.deep_supervision)
        self.seed = getattr(args, "seed", None)
        if self.seed is not None:
            self._seed_everything(int(self.seed))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_dir, self.gt_mask_dir = self._data_dirs(args.data_root)
        train_split = resolve_split_path(Path(args.data_root), "trainval.txt")
        test_split = resolve_split_path(Path(args.data_root), "test.txt")
        self.train_ids = self._load_ids(train_split)
        self.test_ids = self._load_ids(test_split)
        self.initial_sup_dir = args.sup_dir or os.path.join(args.data_root, *SUP_SUBDIR)
        if not os.path.isdir(self.initial_sup_dir):
            raise FileNotFoundError(f"Supervision directory not found: {self.initial_sup_dir}")

        self.input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        test_set = PointSupTestSet(
            self.image_dir,
            self.gt_mask_dir,
            self.test_ids,
            base_size=args.base_size,
            transform=self.input_transform,
        )
        self.test_loader = DataLoader(
            test_set,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=False,
        )

        model = build_network(
            name=args.network,
            channel_size=args.channel_size,
            backbone=args.backbone,
            deep_supervision=self.args.deep_supervision,
        )
        self._initialize_weights(model)
        gpu_ids = list(range(torch.cuda.device_count()))
        if self.device.type == "cuda" and len(gpu_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)
            print(f"  Using DataParallel on GPUs: {gpu_ids}")
        self.model = model.to(self.device)
        self.model_raw = model.module if hasattr(model, "module") else model

        if args.loss == "grsr":
            self.loss_fn = GRSRContinuousLoss(
                w_fn=args.w_fn,
                w_fp=args.w_fp,
                focusing_gamma=args.focusing_gamma,
                aiou_alpha=args.aiou_alpha,
                temperature=args.temperature,
                aiou_band=getattr(args, "aiou_band", 0),
                lambda_a=getattr(args, "lambda_a", 1.0),
                lambda_s=getattr(args, "lambda_s", 0.5),
            )
        else:
            self.loss_fn = baseline_continuous_loss

        tag = args.save_tag.strip() or f"grsr_{args.network}"
        self.save_dir = os.path.join(str(REPO_ROOT), "results", tag)
        os.makedirs(self.save_dir, exist_ok=True)
        self._write_run_config(train_split, test_split)
        self.iteration_metrics = []

        print("[GRSR stage 2] Data-Driven Iterative Label Refinement")
        print(
            f"  paper-fixed: optimizer=Adam, lr={args.lr}, batch={args.batch_size}, "
            f"epochs/iteration={args.epochs}, iterations={args.iterations}, "
            f"gamma_high={args.gamma_high}, gamma_low={args.gamma_low}"
        )
        print(
            f"  optimization space: w_fn={args.w_fn}, w_fp={args.w_fp}, "
            f"focusing_gamma={args.focusing_gamma}, aiou_alpha={args.aiou_alpha}, "
            f"temperature={args.temperature}"
        )
        print(f"  backbone={args.network}, train={len(self.train_ids)}, test={len(self.test_ids)}")
        print(f"  initial supervision={self.initial_sup_dir}")
        print(f"  output={self.save_dir}")
        print(f"  device={self.device}")

    @staticmethod
    def _seed_everything(seed):
        """Seed all local RNGs when a seed is explicitly requested."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # These flags only affect runs that opt into --seed and make the
        # resulting manifest's reproducibility claim meaningful.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def _git_metadata():
        def run_git(*args):
            try:
                return subprocess.check_output(
                    ["git", "-C", str(REPO_ROOT), *args],
                    stderr=subprocess.DEVNULL,
                    text=True,
                ).strip()
            except (OSError, subprocess.CalledProcessError):
                return None

        return {
            "commit": run_git("rev-parse", "HEAD"),
            "status": run_git("status", "--porcelain"),
        }

    def _write_run_config(self, train_split, test_split):
        """Persist the exact run configuration beside checkpoints."""
        try:
            torch_version = torch.__version__
        except AttributeError:
            torch_version = None
        payload = {
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "args": vars(self.args),
            "seed": self.seed,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch_version,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
            "cuda_device_names": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ] if torch.cuda.is_available() else [],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "repo_root": str(REPO_ROOT),
            "git": self._git_metadata(),
            "data_root": str(Path(self.args.data_root).expanduser().resolve()),
            "image_dir": str(Path(self.image_dir).resolve()),
            "ground_truth_dir": str(Path(self.gt_mask_dir).resolve()),
            "train_split": str(Path(train_split).resolve()),
            "test_split": str(Path(test_split).resolve()),
            "train_count": len(self.train_ids),
            "test_count": len(self.test_ids),
        }
        with open(os.path.join(self.save_dir, "run_config.json"), "w") as handle:
            json.dump(payload, handle, indent=2, default=str)

    @staticmethod
    def _data_dirs(data_root):
        if os.path.isdir(os.path.join(data_root, "IRSTD1k_Img")):
            return (
                os.path.join(data_root, "IRSTD1k_Img"),
                os.path.join(data_root, "IRSTD1k_Label"),
            )
        return os.path.join(data_root, "images"), os.path.join(data_root, "masks")

    @staticmethod
    def _load_ids(path):
        with open(path) as handle:
            ids = []
            for line in handle:
                value = line.strip()
                if not value:
                    continue
                # Official split files sometimes store filenames while the
                # pipeline dataset classes append ``.png`` themselves.
                if value.lower().endswith(".png"):
                    value = value[:-4]
                ids.append(value)
            return ids

    @staticmethod
    def _initialize_weights(model):
        from torch.nn import init

        def initialize(module):
            if hasattr(module, "weight") and module.weight is not None and module.weight.dim() > 1:
                init.xavier_normal_(module.weight.data)

        model.apply(initialize)

    def _set_train_supervision(self, supervision_dir):
        train_set = PointSupTrainSet(
            self.image_dir,
            supervision_dir,
            self.train_ids,
            base_size=self.args.base_size,
            crop_size=self.args.crop_size,
            transform=self.input_transform,
        )
        self.train_loader = DataLoader(
            train_set,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            drop_last=True,
        )

    def _reset_optimizer(self, lr=None):
        lr = self.args.lr if lr is None else lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.args.epochs, eta_min=0.0
        )

    def train_one_epoch(self, iteration, epoch):
        self.model.train()
        losses = []
        progress = tqdm(
            self.train_loader,
            desc=f"Iteration {iteration} Epoch {epoch + 1}/{self.args.epochs}",
            ncols=96,
        )
        for data, target in progress:
            data, target = data.to(self.device), target.to(self.device)
            predictions = self.model(data)
            if isinstance(predictions, (list, tuple)):
                loss = 0.0
                for prediction in predictions:
                    resized_target = nn.functional.interpolate(
                        target,
                        size=prediction.shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    loss = loss + self.loss_fn(prediction, resized_target)
                # Average deep-supervision branches, matching DNANet training.
                # Summing them changes the effective learning rate by 4x.
                loss = loss / len(predictions)
            else:
                loss = self.loss_fn(predictions, target)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            losses.append(float(loss.item()))
            progress.set_postfix(loss=f"{loss.item():.4f}")
        self.scheduler.step()
        return float(np.mean(losses))

    @torch.no_grad()
    def evaluate(self, iteration):
        self.model.eval()
        iou_metric = mIoU(1)
        pdfa_metric = PD_FA(1, 10)
        roc_metric = ROCMetric(1, 10)
        for data, labels in tqdm(self.test_loader, desc=f"Evaluate iteration {iteration}", ncols=96):
            data, labels = data.to(self.device), labels.to(self.device)
            output = self.model(data)
            prediction = output[-1] if isinstance(output, (list, tuple)) else output
            iou_metric.update(prediction, labels)
            roc_metric.update(prediction, labels)
            for index in range(prediction.shape[0]):
                pdfa_metric.update(
                    torch.sigmoid(prediction[index, 0]) * 255,
                    labels[index, 0] * 255,
                )

        _, iou = iou_metric.get()
        false_alarm, probability_detection = pdfa_metric.get(len(self.test_ids))
        tpr, fpr, recall, precision = roc_metric.get()
        metric = {
            "iteration": iteration,
            "iou": float(iou),
            "pd": float(probability_detection[5]),
            "fa": float(false_alarm[5]),
            "PD": probability_detection,
            "FA": false_alarm,
            "TPR": tpr,
            "FPR": fpr,
            "recall": recall,
            "precision": precision,
        }
        print(
            f"Iteration {iteration}: IoU={100 * metric['iou']:.2f}, "
            f"Pd={100 * metric['pd']:.2f}, Fa={1e6 * metric['fa']:.2f}e-6"
        )
        return metric

    @torch.no_grad()
    def update_pseudo_masks(self, current_dir, next_dir):
        """Update pseudo masks with confidence-gated predictions.

        ``union`` is the manuscript-compatible monotone update.  ``decay`` is
        an optional conservative variant for experiments: high-confidence
        foreground responses are still merged by maximum, while low-confidence
        predictions gradually attenuate only weak existing values.  Strong
        pseudo-label cores are protected, so the update cannot erase the
        point-centred seed in one step.
        """
        os.makedirs(next_dir, exist_ok=True)
        self.model.eval()
        for name in tqdm(self.train_ids, desc="Pseudo-mask update", ncols=96):
            image = Image.open(os.path.join(self.image_dir, name + ".png")).convert("RGB")
            image_small = image.resize((self.args.base_size, self.args.base_size), Image.BILINEAR)
            tensor = self.input_transform(np.array(image_small, dtype=np.uint8))
            output = self.model(tensor.unsqueeze(0).to(self.device))
            logits = output[-1] if isinstance(output, (list, tuple)) else output

            current_path = os.path.join(current_dir, name + ".png")
            current = np.array(Image.open(current_path).convert("L"), dtype=np.float32) / 255.0
            probability = torch.sigmoid(logits)
            probability = nn.functional.interpolate(
                probability,
                size=current.shape,
                mode="bilinear",
                align_corners=False,
            )[0, 0].cpu().numpy()
            gate_radius = getattr(self.args, "point_gate", 0)
            if gate_radius > 0:
                probability = self._point_gate(name, probability, gate_radius)
            mode = getattr(self.args, "update_mode", "union")
            prior = None
            if mode == "anchor":
                prior_path = os.path.join(self.initial_sup_dir, name + ".png")
                prior = np.array(Image.open(prior_path).convert("L"), dtype=np.float32) / 255.0
                if prior.shape != current.shape:
                    prior = np.array(
                        Image.fromarray((prior * 255).astype(np.uint8)).resize(
                            (current.shape[1], current.shape[0]), Image.BILINEAR
                        ),
                        dtype=np.float32,
                    ) / 255.0
            updated = apply_confidence_update(
                current,
                probability,
                gamma_high=getattr(self.args, "gamma_high", 0.9),
                gamma_low=getattr(self.args, "gamma_low", 0.2),
                update_mode=mode,
                low_decay=getattr(self.args, "low_decay", 0.25),
                protection_threshold=getattr(
                    self.args, "protection_threshold", 0.8
                ),
                prior=prior,
                anchor_beta=getattr(self.args, "anchor_beta", 0.25),
                prior_core=getattr(self.args, "prior_core", 0.5),
            )
            Image.fromarray((np.clip(updated, 0.0, 1.0) * 255).astype(np.uint8)).save(
                os.path.join(next_dir, name + ".png")
            )

    def _point_gate(self, name, probability, radius):
        """Keep only confident components that lie within ``radius`` of an annotated point.

        A confident response with no annotated point nearby cannot be a labelled
        target under point supervision, so it is treated as confident background
        before the update instead of being written into the next pseudo label.
        """
        point_path = os.path.join(
            self.args.data_root, getattr(self.args, "point_dir", "masks_centroid"), name + ".png"
        )
        points = np.array(Image.open(point_path).convert("L")) > 127
        coords = np.argwhere(points).astype(np.float32)
        seed = np.zeros(probability.shape, dtype=bool)
        if len(coords):
            scale = np.array(probability.shape, dtype=np.float32) / np.array(points.shape, dtype=np.float32)
            coords = np.clip(np.round(coords * scale).astype(int), 0, np.array(probability.shape) - 1)
            seed[coords[:, 0], coords[:, 1]] = True
        near = ndi.distance_transform_edt(~seed) <= radius
        labels, count = ndi.label(probability > getattr(self.args, "gamma_high", 0.9))
        if count == 0:
            return probability
        keep = np.unique(labels[near & (labels > 0)])
        outside = (labels > 0) & ~np.isin(labels, keep)
        gated = probability.copy()
        gated[outside] = 0.0
        return gated

    def _write_results(self):
        lines = ["GRSR iterative evaluation (paper units)\n"]
        lines.append("Iteration  IoU(1e-2)  Pd(1e-2)  Fa(1e-6)\n")
        for metric in self.iteration_metrics:
            lines.append(
                f"{metric['iteration']:9d}  {100 * metric['iou']:9.2f}  "
                f"{100 * metric['pd']:8.2f}  {1e6 * metric['fa']:8.2f}\n"
            )

        final = self.iteration_metrics[-1]
        lines.append("\nFinal PD/FA thresholds\n")
        lines.append("Bin       Pd              Fa\n")
        for index, (pd_value, fa_value) in enumerate(zip(final["PD"], final["FA"])):
            lines.append(f"{index:3d}  {pd_value:8.4f}  {fa_value:14.6e}\n")
        lines.append("\nFinal ROC thresholds\n")
        lines.append("Bin      TPR             FPR    Recall  Precision\n")
        for index in range(len(final["TPR"])):
            lines.append(
                f"{index:3d}  {final['TPR'][index]:8.4f}  {final['FPR'][index]:14.6e}  "
                f"{final['recall'][index]:8.4f}  {final['precision'][index]:9.4f}\n"
            )

        output_path = os.path.join(self.save_dir, "final_eval.txt")
        with open(output_path, "w") as handle:
            handle.writelines(lines)
        print(f"Results saved to {output_path}")

    def _reinitialize_model(self):
        for module in self.model_raw.modules():
            if module is not self.model_raw and hasattr(module, "reset_parameters"):
                module.reset_parameters()
        self._initialize_weights(self.model_raw)

    def run(self):
        current_supervision = self.initial_sup_dir
        start = getattr(self.args, "start_iteration", 1)
        if start > 1:
            state = torch.load(self.args.resume_checkpoint, map_location=self.device)
            self.model_raw.load_state_dict(state)
            if self.args.resume_sup_dir:
                current_supervision = self.args.resume_sup_dir
            else:
                # Rebuild the round-2 labels from the round-1 detector with the current update settings.
                if start != 2:
                    raise ValueError("label regeneration on resume is only defined for start_iteration=2")
                current_supervision = os.path.join(self.save_dir, "pseudo_masks", "iteration_2")
                self.update_pseudo_masks(self.initial_sup_dir, current_supervision)
            print(f"  resumed round {start} from {self.args.resume_checkpoint} with {current_supervision}")
        for iteration in range(start, self.args.iterations + 1):
            self._set_train_supervision(current_supervision)
            if iteration > 1 and getattr(self.args, "reinit_each_round", False):
                self._reinitialize_model()
            lr = self.args.lr * (getattr(self.args, "round_lr_scale", 1.0) if iteration > 1 else 1.0)
            self._reset_optimizer(lr)
            for epoch in range(self.args.epochs):
                loss = self.train_one_epoch(iteration, epoch)
                print(f"Iteration {iteration}, epoch {epoch + 1}: loss={loss:.4f}")

            checkpoint = os.path.join(self.save_dir, f"iteration_{iteration}.pth")
            torch.save(self.model_raw.state_dict(), checkpoint)
            self.iteration_metrics.append(self.evaluate(iteration))

            if iteration < self.args.iterations:
                next_supervision = os.path.join(
                    self.save_dir, "pseudo_masks", f"iteration_{iteration + 1}"
                )
                self.update_pseudo_masks(current_supervision, next_supervision)
                current_supervision = next_supervision

        torch.save(self.model_raw.state_dict(), os.path.join(self.save_dir, "final_model.pth"))
        self._write_results()


def parse_args():
    parser = argparse.ArgumentParser(
        description="GRSR training, iterative pseudo-mask refinement, and evaluation"
    )
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--sup_dir", default="")
    parser.add_argument("--network", default="dnanet", choices=available_networks())
    parser.add_argument("--loss", default="grsr", choices=("grsr", "baseline"))

    fixed = parser.add_argument_group("Paper-fixed training settings")
    fixed.add_argument("--epochs", type=int, default=500)
    fixed.add_argument("--iterations", type=int, default=3)
    fixed.add_argument("--batch_size", type=int, default=4)
    fixed.add_argument("--lr", type=float, default=1e-3)
    fixed.add_argument("--gamma_high", type=float, default=0.9)
    fixed.add_argument("--gamma_low", type=float, default=0.2)
    fixed.add_argument(
        "--update_mode",
        choices=("union", "decay", "high_only", "anchor"),
        default="union",
        help="Pseudo-mask update rule; union is manuscript-compatible",
    )
    fixed.add_argument(
        "--low_decay",
        type=float,
        default=0.25,
        help="Decay step for weak low-confidence values in decay mode",
    )
    fixed.add_argument(
        "--protection_threshold",
        type=float,
        default=0.8,
        help="Protect current pseudo-label values at or above this level",
    )
    fixed.add_argument(
        "--anchor_beta",
        type=float,
        default=0.25,
        help="Share of the initial prior G_0 kept at every pixel in anchor mode",
    )
    fixed.add_argument(
        "--prior_core",
        type=float,
        default=0.5,
        help="Prior level above which background evidence may not lower the label in anchor mode",
    )

    tuning = parser.add_argument_group("Optimization space (not fixed in the manuscript)")
    tuning.add_argument("--w_fn", type=float, default=3.0)
    tuning.add_argument("--w_fp", type=float, default=1.0)
    tuning.add_argument("--focusing_gamma", type=float, default=2.0)
    tuning.add_argument("--aiou_alpha", type=float, default=0.3)
    tuning.add_argument(
        "--aiou_band",
        type=int,
        default=0,
        help="Band in pixels around the target where false positives are discounted; 0 keeps the global discount",
    )
    tuning.add_argument("--temperature", type=float, default=0.1)
    tuning.add_argument("--lambda_a", type=float, default=1.0, help="Weight of the asymmetric SoftIoU term")
    tuning.add_argument("--lambda_s", type=float, default=0.5, help="Weight of the saliency peak attention term")
    tuning.add_argument("--base_size", type=int, default=256)
    tuning.add_argument("--crop_size", type=int, default=256)
    tuning.add_argument("--num_workers", type=int, default=4)

    rounds = parser.add_argument_group("Round schedule and point-guided update (defaults reproduce the original behaviour)")
    rounds.add_argument("--point_gate", type=int, default=0,
                        help="Radius in pixels; confident components without an annotated point within it are treated as background in the update. 0 disables")
    rounds.add_argument("--point_dir", default="masks_centroid")
    rounds.add_argument("--round_lr_scale", type=float, default=1.0,
                        help="Learning-rate multiplier for rounds after the first")
    rounds.add_argument("--reinit_each_round", type=_as_bool, default=False)
    rounds.add_argument("--start_iteration", type=int, default=1)
    rounds.add_argument("--resume_checkpoint", default="")
    rounds.add_argument("--resume_sup_dir", default="")

    parser.add_argument("--channel_size", default="three")
    parser.add_argument("--backbone", default="resnet_18")
    parser.add_argument("--deep_supervision", type=_as_bool, default=True)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--save_tag", default="")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed; when set, the value and environment are saved in run_config.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = arguments.gpus
    PointSupTrainer(arguments).run()
