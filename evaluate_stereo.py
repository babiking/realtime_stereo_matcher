from __future__ import print_function, division
import sys
import time
import json
import logging
import numpy as np
import torch
from tqdm import tqdm
from model import build_model
import dataset.stereo_datasets as datasets
from dataset.input_padder import InputPadder
import gflags

gflags.DEFINE_string(
    "exp_config_json", "configure/exp_config.json", "experiment configure json file"
)
gflags.DEFINE_string(
    "model_chkpt_file",
    "experiments/BASE_STEREO_NET/checkpoints/BASE_STEREO_NET-epoch-100000.pth.gz",
    "model checkpont file",
)

autocast = torch.cuda.amp.autocast


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def validate_eth3d(model, mixed_prec=False):
    """Peform validation using the ETH3D (train) split"""
    model.eval()
    aug_params = {}
    val_dataset = datasets.ETH3D(aug_params)

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            flow_pr = model(image1, image2)[-1]
        flow_pr = padder.unpad(flow_pr.float()).cpu().squeeze(0)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt) ** 2, dim=0).sqrt()

        epe_flattened = epe.flatten()
        val = valid_gt.flatten() >= 0.5
        out = epe_flattened > 1.0
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        logging.info(
            f"ETH3D {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}"
        )

        if image_epe > 80.0:
            continue

        epe_list.append(image_epe)
        out_list.append(image_out)

    epe_list = np.array(epe_list)
    out_list = np.array(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    print("Validation ETH3D: EPE %f, D1 %f" % (epe, d1))
    return {"eth3d-epe": epe, "eth3d-d1": d1}


@torch.no_grad()
def validate_kitti(model, mixed_prec=False):
    """Peform validation using the KITTI-2015 (train) split"""
    model.eval()
    aug_params = {}
    val_dataset = datasets.KITTI(aug_params, image_set="training")
    torch.backends.cudnn.benchmark = True

    out_list, epe_list, elapsed_list = [], [], []
    for val_id in range(len(val_dataset)):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            start = time.time()
            flow_pr = model(image1, image2)[-1]
            end = time.time()

        if val_id > 50:
            elapsed_list.append(end - start)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt) ** 2, dim=0).sqrt()

        epe_flattened = epe.flatten()
        val = valid_gt.flatten() >= 0.5

        out = epe_flattened > 1.0
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        if val_id < 9 or (val_id + 1) % 10 == 0:
            logging.info(
                f"KITTI Iter {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}. Runtime: {format(end-start, '.3f')}s ({format(1/(end-start), '.2f')}-FPS)"
            )
        epe_list.append(epe_flattened[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    avg_runtime = np.mean(elapsed_list)

    print(
        f"Validation KITTI: EPE {epe}, D1 {d1}, {format(1/avg_runtime, '.2f')}-FPS ({format(avg_runtime, '.3f')}s)"
    )
    return {"kitti-epe": epe, "kitti-d1": d1}


@torch.no_grad()
def validate_things(model, mixed_prec=False):
    """Peform validation using the FlyingThings3D (TEST) split"""
    model.eval()
    val_dataset = datasets.SceneFlowDatasets(
        dstype="frames_finalpass", things_test=True
    )

    out_list, epe_list = [], []
    for val_id in tqdm(range(len(val_dataset))):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            flow_pr = model(image1, image2)[-1]
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt) ** 2, dim=0).sqrt()

        epe = epe.flatten()
        val = (valid_gt.flatten() >= 0.5) & (flow_gt.abs().flatten() < 192)

        out = epe > 1.0
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    print("Validation FlyingThings: %f, %f" % (epe, d1))
    return {"things-epe": epe, "things-d1": d1}


@torch.no_grad()
def validate_middlebury(model, split="F", mixed_prec=False):
    """Peform validation using the Middlebury-V3 dataset"""
    model.eval()
    aug_params = {}
    val_dataset = datasets.Middlebury(aug_params, split=split)

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        (imageL_file, _, _), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            flow_pr = model(image1, image2)[-1]
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt) ** 2, dim=0).sqrt()

        epe_flattened = epe.flatten()
        val = (valid_gt.reshape(-1) >= -0.5) & (flow_gt[0].reshape(-1) > -1000)

        out = epe_flattened > 1.0
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        logging.info(
            f"Middlebury Iter {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}"
        )
        epe_list.append(image_epe)
        out_list.append(image_out)

    epe_list = np.array(epe_list)
    out_list = np.array(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    print(f"Validation Middlebury{split}: EPE {epe}, D1 {d1}")
    return {f"middlebury{split}-epe": epe, f"middlebury{split}-d1": d1}


def main():
    FLAGS = gflags.FLAGS
    FLAGS(sys.argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    )

    exp_config = json.load(open(FLAGS.exp_config_json, "r"))

    model = torch.nn.DataParallel(build_model(exp_config["model"])).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    model.cuda()
    model.eval()

    assert FLAGS.model_chkpt_file.endswith(".pth") or FLAGS.model_chkpt_file.endswith(
        ".pth.gz"
    )
    logging.info(f"Loading checkpoint: {FLAGS.model_chkpt_file}...")
    checkpoint = torch.load(FLAGS.model_chkpt_file)
    model.load_state_dict(checkpoint, strict=True)
    logging.info(f"Done loading checkpoint.")

    print(
        f"The model has {format(count_parameters(model)/1e6, '.4f')}M learnable parameters."
    )

    # The CUDA implementations of the correlation volume prevent half-precision
    # rounding errors in the correlation lookup. This allows us to use mixed precision
    # in the entire forward pass, not just in the GRUs & feature extractors.
    use_mixed_precision = exp_config["model"].get("mixed_precision", True)

    for dataset in exp_config["test"]["datasets"]:
        if dataset == "eth3d":
            validate_eth3d(model, mixed_prec=use_mixed_precision)

        elif dataset == "kitti":
            validate_kitti(model, mixed_prec=use_mixed_precision)

        elif dataset in ([f"middlebury_{s}" for s in "FHQ"] + ["middlebury_2014"]):
            validate_middlebury(
                model,
                split=dataset.split("_")[-1],
                mixed_prec=use_mixed_precision,
            )

        elif dataset == "things":
            validate_things(model, mixed_prec=use_mixed_precision)


if __name__ == "__main__":
    main()
