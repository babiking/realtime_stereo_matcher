from __future__ import print_function, division

import os
import sys
import json
import logging
import numpy as np
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model.mobile_raft_stereo import MobileRaftStereoModel
import dataset.stereo_datasets as datasets

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            pass


import gflags

gflags.DEFINE_string(
    "exp_config_json",
    "configure/exp_config.json",
    "experiment configure json file",
)


def sequence_loss(flow_preds, flow_gt, valid, loss_gamma=0.9, max_flow=700):
    """Loss function defined over sequence of flow predictions"""

    n_predictions = len(flow_preds)
    assert n_predictions >= 1
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()

    # exclude extremly large displacements
    valid = ((valid >= 0.5) & (mag < max_flow)).unsqueeze(1)
    assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
    assert not torch.isinf(flow_gt[valid.bool()]).any()

    for i in range(n_predictions):
        assert (
            not torch.isnan(flow_preds[i]).any()
            and not torch.isinf(flow_preds[i]).any()
        )
        # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
        adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions))
        i_weight = adjusted_loss_gamma ** (n_predictions - i)
        i_loss = (flow_preds[i] - flow_gt).abs()
        assert i_loss.shape == valid.shape, [
            i_loss.shape,
            valid.shape,
            flow_gt.shape,
            flow_preds[i].shape,
        ]
        flow_loss += i_weight * i_loss[valid.bool()].mean()

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        "epe": epe.mean().item(),
        "1px": (epe < 1).float().mean().item(),
        "3px": (epe < 3).float().mean().item(),
        "5px": (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


def fetch_optimizer(exp_config, model):
    """Create the optimizer and learning rate scheduler"""
    optimizer = optim.AdamW(
        model.parameters(),
        lr=exp_config["train"]["learn_rate"],
        weight_decay=exp_config["train"]["weight_decay"],
        eps=1e-8,
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        exp_config["train"]["learn_rate"],
        exp_config["train"]["num_of_steps"] + 100,
        pct_start=0.01,
        cycle_momentum=False,
        anneal_strategy="linear",
    )

    return optimizer, scheduler


class Logger:
    SUM_FREQ = 100

    def __init__(self, model, scheduler, log_dir=None):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.log_dir = log_dir
        self.writer = SummaryWriter(
            log_dir="runs" if self.log_dir is None else self.log_dir
        )

    def _print_training_status(self):
        metrics_data = [
            self.running_loss[k] / Logger.SUM_FREQ
            for k in sorted(self.running_loss.keys())
        ]
        training_str = "[{:6d}, {:10.7f}] ".format(
            self.total_steps + 1, self.scheduler.get_last_lr()[0]
        )
        metrics_str = ("{:10.4f}, " * len(metrics_data)).format(*metrics_data)

        # print the training status
        logging.info(
            f"Training Metrics ({self.total_steps}): {training_str + metrics_str}"
        )

        if self.writer is None:
            self.writer = SummaryWriter(
                log_dir="runs" if self.log_dir is None else self.log_dir
            )

        for k in self.running_loss:
            self.writer.add_scalar(
                k, self.running_loss[k] / Logger.SUM_FREQ, self.total_steps
            )
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ - 1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter(
                log_dir="runs" if self.log_dir is None else self.log_dir
            )

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

def initialize(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def train(exp_config):
    model = nn.DataParallel(MobileRaftStereoModel(**exp_config["model"]))
    logging.info(f"Model parameter count (pytorch): {count_parameters(model)}.")

    train_loader = datasets.fetch_dataloader(exp_config)
    optimizer, scheduler = fetch_optimizer(exp_config, model)
    total_steps = 0
    logger = Logger(model, scheduler, log_dir=os.path.join(exp_config["path"], "runs"))

    restore_ckpt = exp_config["train"]["restore_checkpoint"]
    if restore_ckpt is not None and len(restore_ckpt) > 0:
        assert restore_ckpt.endswith(".pth") or restore_ckpt.endswith(".pth.gz")
        logging.info(f"Model loading checkpoint from {restore_ckpt}...")
        model.load_state_dict(torch.load(restore_ckpt), strict=True)
        logging.info(f"Done loading checkpoint.")

    model.cuda()
    model.train()
    initialize(model.module)

    scaler = GradScaler(enabled=exp_config["model"]["mixed_precision"])

    should_keep_training = True
    global_batch_num = 0
    while should_keep_training:
        for i_batch, (_, *data_blob) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            image1, image2, flow, valid = [x.cuda() for x in data_blob]

            assert model.training
            flow_predictions = model(image1, image2)
            assert model.training

            loss, metrics = sequence_loss([flow_predictions[-1]], flow, valid)
            logger.writer.add_scalar("live_loss", loss.item(), global_batch_num)
            logger.writer.add_scalar(
                f"learning_rate", optimizer.param_groups[0]["lr"], global_batch_num
            )
            global_batch_num += 1
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(metrics)

            total_steps += 1

            if total_steps > exp_config["train"]["num_of_steps"]:
                should_keep_training = False
                break

            if total_steps % exp_config["train"]["save_checkpoint_frequency"] == 0:
                exp_name = exp_config["name"]
                exp_path = exp_config["path"]
                save_ckpt_file = os.path.join(
                    exp_path, f"checkpoints/{exp_name}-epoch-{total_steps}.pth.gz"
                )
                os.makedirs(os.path.dirname(save_ckpt_file), exist_ok=True)
                logging.info(f"Saving file {save_ckpt_file}...")
                torch.save(model.state_dict(), save_ckpt_file)

    logging.info("FINISHED TRAINING!")
    logger.close()
    final_ckpt_file = os.path.join(
        exp_path, f"checkpoints/{exp_name}-epoch-{total_steps}.pth.gz"
    )
    torch.save(model.state_dict(), final_ckpt_file)
    return final_ckpt_file


def main():
    FLAGS = gflags.FLAGS
    FLAGS(sys.argv)

    torch.manual_seed(1234)
    np.random.seed(1234)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    )

    exp_config = json.load(open(FLAGS.exp_config_json, "r"))
    train(exp_config)


if __name__ == "__main__":
    main()
