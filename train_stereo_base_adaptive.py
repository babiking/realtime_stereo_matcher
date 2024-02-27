from __future__ import print_function, division

import os
import sys
import csv
import json
import logging
import numpy as np
import cv2 as cv
from tools.colorize import colorize_2d_matrix

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from loss import build_loss_function
from loss.loss import get_flow_map_metrics
import dataset.stereo_datasets as datasets
from model.mobile_stereo_base import MobileStereoBase
from torch.cuda.amp import GradScaler


import gflags

gflags.DEFINE_string(
    "experiment",
    "configure/trainer_base_v1_adaptive.json",
    "experiment configure json file",
)
gflags.DEFINE_string(
    "model",
    "configure/stereo_base_net_v1.json",
    "model configure json file"
)


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
    SUM_FREQ = 1

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
        training_str = "[{:6d}, {:10.7f}] ".format(
            self.total_steps + 1, self.scheduler.get_last_lr()[0]
        )
        metrics_str = ", ".join(
            [
                f"{k}:{self.running_loss[k] / Logger.SUM_FREQ:.4f}"
                for k in self.running_loss.keys()
            ]
        )

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
        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % Logger.SUM_FREQ == 0:
            self._print_training_status()
            self.running_loss = {}

        self.total_steps += 1

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


def get_stereo_dataset(data_name):
    raise NotImplementedError


def train(exp_config, model_config):
    model = nn.DataParallel(MobileStereoBase(model_config))
    logging.info(f"Model parameter count (pytorch): {count_parameters(model)}.")

    loss_func = build_loss_function(exp_config["train"]["loss"])

    optimizer, scheduler = fetch_optimizer(exp_config, model)
    total_steps = 0
    logger = Logger(
        model, scheduler, log_dir=os.path.join(exp_config["path"], "runs")
    )

    model.cuda()
    model.train()
    initialize(model.module)

    scaler = GradScaler(enabled=exp_config["model"].get("mixed_precision", False))

    model.load_state_dict(
        torch.load(exp_config["train"]["restore_checkpoint"]), strict=True
    )

    dataloader = datasets.fetch_dataloader(exp_config)

    should_keep_training = True
    global_batch_num = 0
    while should_keep_training:
        for i, (img_files, l_img, r_img, flow, valid) in enumerate(dataloader):
            optimizer.zero_grad()

            l_img = l_img.cuda().unsqueeze(0)
            r_img = r_img.cuda().unsqueeze(0)
            flow = flow.cuda().unsqueeze(0)
            valid = valid.cuda().unsqueeze(0)

            l_disps = model(l_img, r_img)
            loss = loss_func(
                l_img,
                r_img,
                [-1.0 * l_disps[-1]],
                [None],
                l_occ=(valid < 0.5),
                r_occ=None,
            )
            metrics = get_flow_map_metrics(flow, l_disps[-1], valid)
            metrics["loss"] = round(loss.item(), 4)
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


def main():
    FLAGS = gflags.FLAGS
    FLAGS(sys.argv)

    torch.manual_seed(1234)
    np.random.seed(1234)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    )

    exp_config = json.load(open(FLAGS.experiment, "r"))
    model_config = json.load(open(FLAGS.model, "r"))
    train(exp_config, model_config)


if __name__ == "__main__":
    main()
