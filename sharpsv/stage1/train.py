import os
import random

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sharpsv-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/sharpsv-cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

import numpy as np
import pytorch_lightning as pl
import ray
import torch
import torch.backends.cudnn as cudnn
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search import Repeater
from ray.tune.search.hyperopt import HyperOptSearch

from .model import SharpSVLightningModel


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

data_dirs = [
    path for path in os.environ.get("SHARPSV_TRAIN_DIRS", "./workdir").split(os.pathsep) if path
]

logger = TensorBoardLogger(os.environ.get("SHARPSV_LOG_DIR", "./logs"), name="training")

checkpoint_callback = ModelCheckpoint(
    dirpath="./checkpoints_predict/",
    filename="{epoch:02d}-{validation_1_re:.2f}-{validation_0_re:.2f}-{validation_1_pre:.2f}",
    monitor="custom_metric",
    verbose=False,
    save_top_k=1,
    mode="max",
    auto_insert_metric_name=False,
)


def train_tune(config, checkpoint_dir=None, num_epochs=32, num_gpus=1):
    model = SharpSVLightningModel(data_dirs, config)
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        devices=num_gpus,
        accelerator="gpu" if torch.cuda.is_available() and num_gpus > 0 else "cpu",
        check_val_every_n_epoch=1,
        logger=logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model)


def run_tuning(num_samples=-1, num_epochs=120):
    config = {
        "lr": tune.loguniform(1e-8, 1e-3),
        "batch_size": 64,
        "beta1": tune.uniform(0.85, 0.95),
        "beta2": tune.uniform(0.995, 0.9999),
        "weight_decay": tune.uniform(0.0001, 0.01),
    }

    bayesopt = HyperOptSearch(config, metric="custom_metric", mode="max")
    re_search_alg = Repeater(bayesopt, repeat=1)
    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)
    reporter = CLIReporter(metric_columns=["train_loss", "validation_loss", "custom_metric"])

    tune.run(
        tune.with_parameters(train_tune, num_epochs=num_epochs),
        local_dir=os.environ.get("SHARPSV_TUNE_DIR", "./tune_results"),
        resources_per_trial={"cpu": 4, "gpu": 1 if torch.cuda.is_available() else 0},
        num_samples=num_samples,
        metric="custom_metric",
        mode="max",
        scheduler=scheduler,
        progress_reporter=reporter,
        resume=False,
        search_alg=re_search_alg,
        max_failures=-1,
        name="sharpsv_tune",
    )


def main(argv=None):
    seed_everything()
    ray.init(ignore_reinit_error=True)
    run_tuning()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
