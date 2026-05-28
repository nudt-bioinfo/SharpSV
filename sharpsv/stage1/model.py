# -*- coding:utf-8 -*-
import re
import torch.optim as optim
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import glob
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sharpsv-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/sharpsv-cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

import pytorch_lightning as pl
from sklearn.metrics import classification_report, precision_recall_fscore_support
from ray import tune
import pandas as pd
import pysam


# ==========================================
# 1. Core component: CBAM (channel + spatial attention)
# ==========================================
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        # Channel attention.
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

        # Spatial attention with a wider receptive field.
        self.conv_spatial = nn.Conv1d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        # --- Channel Attention ---
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        scale_c = self.sigmoid(out)
        x = x * scale_c

        # --- Spatial Attention ---
        avg_out_s = torch.mean(x, dim=1, keepdim=True)
        max_out_s, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out_s, max_out_s], dim=1)
        scale_s = self.sigmoid(self.conv_spatial(x_cat))

        return x * scale_s


# ==========================================
# 2. Core component: multi-scale convolution
# ==========================================
class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        quarter_c = out_channels // 4

        # Parallel kernels capture SV signals across multiple scales.
        self.branch1 = nn.Conv1d(in_channels, quarter_c, kernel_size=1)
        self.branch2 = nn.Conv1d(in_channels, quarter_c, kernel_size=3, padding=1)
        self.branch3 = nn.Conv1d(in_channels, quarter_c, kernel_size=5, padding=2)
        self.branch4 = nn.Conv1d(in_channels, quarter_c, kernel_size=7, padding=3)

        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        out = torch.cat([b1, b2, b3, b4], dim=1)
        out = self.bn(out)
        return self.relu(out)


# ==========================================
# 3. Stage-1 model architecture
# ==========================================
class AdvancedSVModel(nn.Module):
    def __init__(self, in_channels=9, base_filters=64):
        super().__init__()

        # Stem: multi-scale feature extraction plus attention weighting.
        self.stem = nn.Sequential(
            MultiScaleConv(in_channels, base_filters),
            CBAM(base_filters),
            nn.MaxPool1d(2)  # 1000 -> 500
        )

        # Layer 1: deeper feature extraction.
        self.layer1 = nn.Sequential(
            nn.Conv1d(base_filters, base_filters * 2, 3, padding=1),
            nn.BatchNorm1d(base_filters * 2),
            nn.ReLU(),
            CBAM(base_filters * 2),
            nn.MaxPool1d(2)  # 500 -> 250
        )

        # Layer 2: deeper hierarchical features.
        self.layer2 = nn.Sequential(
            nn.Conv1d(base_filters * 2, base_filters * 4, 3, padding=1),
            nn.BatchNorm1d(base_filters * 4),
            nn.ReLU(),
            CBAM(base_filters * 4),
            nn.MaxPool1d(2)  # 250 -> 125
        )

        # Hybrid pooling.
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Classification head.
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            # Input width equals max-pooled plus avg-pooled features.
            nn.Linear(base_filters * 4 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x shape: (B, 9, 1000)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)

        # Use both the strongest signal and the global context.
        max_feat = self.global_max_pool(x).flatten(1)
        avg_feat = self.global_avg_pool(x).flatten(1)
        feat = torch.cat([max_feat, avg_feat], dim=1)

        logits = self.fc(feat)
        probs = torch.sigmoid(logits)

        # Preserve the legacy three-value return signature.
        return probs, probs, probs


# ==========================================
# 4. Loss function
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.20, gamma=2.0):
        # Alpha balances the highly imbalanced negative-heavy regime.
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
        bce = F.binary_cross_entropy(pred, target, reduction="none")
        pt = target * pred + (1 - target) * (1 - pred)
        focal_weight = (self.alpha * target + (1 - self.alpha) * (1 - target)) * ((1 - pt) ** self.gamma)
        return (focal_weight * bce).mean()


# ==========================================
# 5. Dataset
# ==========================================
class SharpSVDataset(Dataset):
    def __init__(self, data_dirs, chromosomes=None, mode="train", vcf_path=None, verbose=True):
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]

        self.data_files = []
        for data_dir in data_dirs:
            self.data_files.extend(glob.glob(os.path.join(data_dir, "*.npz")))

        self.data_dirs = data_dirs
        self.data = []
        self.labels = []
        self.indices = []
        self.mode = mode
        self.vcf_path = vcf_path
        self.verbose = verbose
        self.loaded_file_count = 0
        self.chromosomes = chromosomes if chromosomes else self._default_chromosomes()
        self._load_data()

    def _default_chromosomes(self):
        if self.mode == "train":
            return [str(i) for i in range(1, 11)] + ['X', 'Y']
        elif self.mode == "test":
            return ([str(i) for i in range(1, 23)]) + ['X', 'Y']
        else:
            raise ValueError("Invalid mode!")

    def _load_data(self):
        if self.verbose:
            print(f"Loading {self.mode} dataset with chromosomes: {self.chromosomes}")

        vcf_intervals = {}
        if hasattr(self, 'vcf_path') and self.vcf_path and os.path.exists(self.vcf_path):
            if self.verbose:
                print(f"Loading VCF from {self.vcf_path}")
            vcf_file = pysam.VariantFile(self.vcf_path)
            for rec in vcf_file.fetch():
                chrom = str(rec.chrom)
                if not chrom.startswith("chr"):
                    chrom = "chr" + chrom
                if chrom not in vcf_intervals:
                    vcf_intervals[chrom] = []
                start = rec.pos
                length = abs(int(rec.info.get('SVLEN', [0])[0]))
                end = start + length
                vcf_intervals[chrom].append((start, end))
            if self.verbose:
                print(f"VCF loaded: {sum(len(v) for v in vcf_intervals.values())} variants.")

        for file in self.data_files:
            filename = os.path.basename(file)
            chrom = filename.split(':')[0]
            if chrom not in self.chromosomes:
                continue

            try:
                data_dict = np.load(file, allow_pickle=True)
                data = data_dict['data']
                indices = data_dict['index']

                # Prediction-only files may omit labels.
                if 'label' in data_dict:
                    labels = data_dict['label']
                else:
                    # Use placeholder labels for inference-only data.
                    labels = np.zeros(len(data), dtype=np.int64)

            except KeyError as e:
                if self.verbose:
                    print(f"Skipping {file}: {e} (Required keys: 'data' and 'index')")
                continue

            if self.mode == "train":
                idx_ones = np.where(labels == 1)[0]
                idx_zeros = np.where(labels == 0)[0]

                if chrom in vcf_intervals and len(idx_ones) > 0:
                    def overlaps(start, vcf_list, window_size=1000):
                        for v_start, v_end in vcf_list:
                            if v_end > start and v_start < start + window_size:
                                return True
                        return False

                    idx_ones = [i for i in idx_ones if overlaps(indices[i], vcf_intervals[chrom])]
                    idx_ones = np.array(idx_ones)

                if len(idx_ones) == 0:
                    sampled_idx = np.random.choice(idx_zeros, min(100, len(idx_zeros)), replace=False)
                else:
                    num_anomalies = len(idx_ones)
                    # Keep a 1:1 class ratio during training.
                    num_normals_to_sample = min(len(idx_zeros), num_anomalies * 1)
                    sampled_normals = np.random.choice(idx_zeros, num_normals_to_sample, replace=False)
                    sampled_idx = np.concatenate([idx_ones, sampled_normals])

                sampled_data = data[sampled_idx]
                sampled_labels = labels[sampled_idx]
                sampled_indices = indices[sampled_idx]
            else:
                sampled_data = data
                sampled_labels = labels
                sampled_indices = indices

            self.data.append(torch.tensor(sampled_data, dtype=torch.float32))
            self.labels.append(torch.tensor(sampled_labels, dtype=torch.long))
            self.indices.extend([(chrom, int(i)) for i in sampled_indices])
            self.loaded_file_count += 1

        if not self.data:
            if self.verbose:
                print(f"No valid data loaded for {self.mode} set.")
            return

        self.data = torch.cat(self.data, dim=0)
        self.labels = torch.cat(self.labels, dim=0)
        assert len(self.indices) == len(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.data[idx]  # (9000,)
        y = self.labels[idx]
        chr_name, pos = self.indices[idx]

        # Restore the flattened 1000x9 window into Conv1d format.
        x = x.view(1000, 9).permute(1, 0)  # Shape: (9, 1000)

        return x, y, (chr_name, pos)


def load_sharpsv_data(data_dir, batch_size=64, train_chromosomes=None, test_chromosomes=None, vcf_path=None):
    train_dataset = SharpSVDataset(data_dir, train_chromosomes, mode="train", vcf_path=vcf_path)
    print(f"Training dataset loaded with {len(train_dataset)} samples.")
    test_dataset = SharpSVDataset(data_dir, test_chromosomes, mode="test")
    print(f"Test dataset loaded with {len(test_dataset)} samples.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, test_loader


# ==========================================
# 6. Lightning module
# ==========================================
class SharpSVLightningModel(pl.LightningModule):

    def __init__(self, path, config, predict_mode=False, prediction_output_csv=None):
        super(SharpSVLightningModel, self).__init__()
        self.save_hyperparameters()
        self.model = AdvancedSVModel(in_channels=9, base_filters=64)
        self.path = path
        self.criterion = FocalLoss(alpha=0.20, gamma=2.0)

        self.lr = config["lr"]
        self.beta1 = config['beta1']
        self.beta2 = config['beta2']
        self.weight_decay = config['weight_decay']
        self.batch_size = config["batch_size"]
        self.best_threshold = 0.40
        self.predict_mode = predict_mode
        self.prediction_output_csv = prediction_output_csv
        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # Only build training loaders outside prediction mode.
        if not self.predict_mode:
            self.train_loader, self.val_loader = load_sharpsv_data(self.path, self.batch_size)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def forward(self, x):
        return self.model(x)

    def training_validation_step(self, batch, batch_idx):
        x, y, index = batch
        # AdvancedSVModel preserves the legacy triple return signature.
        bag_pred, _, _ = self.model(x)
        loss = self.criterion(bag_pred.squeeze(), y.float())
        return loss, y, bag_pred.squeeze(), index

    def training_step(self, batch, batch_idx):
        loss, y, y_hat, index = self.training_validation_step(batch, batch_idx)
        predictions = (y_hat >= self.best_threshold).long()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=self.batch_size)
        self.train_step_outputs.append(
            {
                'y': torch.as_tensor(y).detach().cpu(),
                'y_hat': torch.as_tensor(predictions).detach().cpu(),
            }
        )
        return loss

    def on_train_epoch_end(self):
        if not self.train_step_outputs:
            return

        y, y_hat = [], []
        for out in self.train_step_outputs:
            y.extend(out['y'])
            y_hat.extend(out['y_hat'])
        y = torch.tensor(y).reshape(-1)
        y_hat = torch.tensor(y_hat).reshape(-1)

        unique_labels = set(y.tolist())
        unique_preds = set(y_hat.tolist())

        if len(unique_labels) == 1 and 1 not in unique_labels:
            metric = {"accuracy": 1.0, "macro avg": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0},
                      "0": {"f1-score": 1.0, "precision": 1.0, "recall": 1.0},
                      "1": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0}}
        elif len(unique_preds) == 1 and 1 not in unique_preds:
            metric = classification_report(y, y_hat, output_dict=True, zero_division=0)
        else:
            metric = classification_report(y, y_hat, output_dict=True)

        self.log('train_macro_f1', metric['macro avg']['f1-score'], on_epoch=True, prog_bar=True)
        self.log('train_0_re', metric['0']['recall'], on_epoch=True, prog_bar=True)
        self.log('train_1_re', metric['1']['recall'] if '1' in metric else 0.0, on_epoch=True, prog_bar=True)
        self.train_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        loss, y, y_hat, index = self.training_validation_step(batch, batch_idx)
        self.log('validation_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=self.batch_size)
        self.validation_step_outputs.append({'y': y, 'y_hat': y_hat, 'index': index})
        return loss


    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return

        all_y, all_probs, index_list = [], [], []

        # Pass 1: collect scores and indices.
        for out in self.validation_step_outputs:
            all_y.extend(torch.as_tensor(out['y']).cpu().numpy().reshape(-1).tolist())
            yh = out['y_hat']
            if isinstance(yh, torch.Tensor):
                all_probs.extend(yh.detach().cpu().numpy().reshape(-1).tolist())
            else:
                all_probs.extend(np.asarray(yh).reshape(-1).tolist())
            index_list.append(out['index'])

        all_y = np.array(all_y, dtype=int)
        all_probs = np.array(all_probs, dtype=float)

        if all_probs.size == 0:
            print("No prediction scores were produced for the current dataset.")
            if self.predict_mode and self.prediction_output_csv:
                output_path = Path(self.prediction_output_csv)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(columns=["chr", "position", "score"]).to_csv(output_path, index=False)
            self.validation_step_outputs.clear()
            return

        # Pass 2: compute the dynamic top-10% threshold in prediction mode.
        if hasattr(self, "predict_mode") and self.predict_mode:
            dynamic_threshold = np.percentile(all_probs, 90)
            self.best_threshold = float(dynamic_threshold)
            print(f"[Top 10% Mode] Dynamic threshold set to: {self.best_threshold:.4f}")
        else:
            # Training mode still performs threshold search.
            candidate_thresholds = np.linspace(0.01, 0.99, 99)
            best_t, best_metric = self.best_threshold, -1.0
            for t in candidate_thresholds:
                p, r, _, _ = precision_recall_fscore_support(all_y, (all_probs >= t).astype(int), labels=[0, 1],
                                                             zero_division=0)
                cm = 0.5 * r[1] + 0.5 * r[0]
                if cm > best_metric:
                    best_metric, best_t = cm, t
            self.best_threshold = float(best_t)

        # Pass 3: save outputs using the updated threshold.
        final_preds = (all_probs >= self.best_threshold).astype(int)
        metric = classification_report(all_y, final_preds, output_dict=True, zero_division=0)

        r1 = metric['1']['recall'] if '1' in metric else 0.0
        p1 = metric['1']['precision'] if '1' in metric else 0.0
        custom_metric = 0.5 * r1 + 0.5 * metric['0']['recall']

        self.log('validation_1_re', r1, on_epoch=True, prog_bar=True)
        self.log('validation_1_pre', p1, on_epoch=True, prog_bar=True)
        self.log('custom_metric', custom_metric, on_epoch=True, prog_bar=True)

        # Report Tune metrics only when a Tune session is active.
        try:
            from ray import tune
            if tune.is_session_enabled():
                tune.report(custom_metric=custom_metric)
        except:
            pass

        positive_records = []
        ptr = 0
        for idx_field in index_list:
            if isinstance(idx_field, (list, tuple)) and len(idx_field) == 2 and isinstance(idx_field[0], (list, tuple)):
                batch_len = len(idx_field[0])
            else:
                try:
                    batch_len = len(idx_field)
                except:
                    batch_len = 1

            batch_probs = all_probs[ptr: ptr + batch_len]
            batch_preds = (batch_probs >= self.best_threshold).astype(int)
            batch_pos_idx = np.where(batch_preds == 1)[0].tolist()

            if isinstance(idx_field, (list, tuple)) and len(idx_field) == 2 and isinstance(idx_field[0], (list, tuple)):
                chr_list, pos_tensor = idx_field
                for i in batch_pos_idx:
                    positive_records.append(
                        {
                            "chr": chr_list[i],
                            "position": int(pos_tensor[i]),
                            "score": float(batch_probs[i]),
                        }
                    )
            else:
                for i in batch_pos_idx:
                    chrom, position = idx_field[i]
                    positive_records.append(
                        {
                            "chr": chrom,
                            "position": int(position),
                            "score": float(batch_probs[i]),
                        }
                    )
            ptr += batch_len

        if self.predict_mode and self.prediction_output_csv:
            output_path = Path(self.prediction_output_csv)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(positive_records, columns=["chr", "position", "score"]).to_csv(output_path, index=False)
            return

        if not hasattr(self, 'best_custom_metric'):
            self.best_custom_metric = -float("inf")

        if custom_metric > self.best_custom_metric:
            self.best_custom_metric = float(custom_metric)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        loss, y, y_hat, index = self.training_validation_step(batch, batch_idx)
        self.test_step_outputs.append(
            {
                'y': torch.as_tensor(y).detach().cpu(),
                'y_hat': torch.as_tensor(y_hat).detach().cpu(),
            }
        )
        return loss

    def on_test_epoch_end(self):
        torch.save(self.test_step_outputs, "result.pt")
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        lr = float(self.lr.sample()) if hasattr(self.lr, "sample") else float(self.lr)
        beta1 = float(self.beta1.sample()) if hasattr(self.beta1, "sample") else float(self.beta1)
        beta2 = float(self.beta2.sample()) if hasattr(self.beta2, "sample") else float(self.beta2)
        weight_decay = float(self.weight_decay.sample()) if hasattr(self.weight_decay, "sample") else float(
            self.weight_decay)
        opt_e = torch.optim.Adam(self.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
        return opt_e


SVDataset = SharpSVDataset
SVLightningModel = SharpSVLightningModel
load_data = load_sharpsv_data



















