import os
import sys
import glob
import json
import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
from tqdm import tqdm

import gflags

gflags.DEFINE_string(
    "data_path",
    "/mnt/data/workspace/datasets/MyRealsense/20240226_fabric_0000",
    "dataset path",
)


def warp_by_flow_map(image, flow):
    """
    warp image according to stereo flow map (i.e. disparity map)

    Args:
        [1] image, N x C x H x W, original image or feature map
        [2] flow,  N x 1 x H x W or N x 2 x H x W, flow map

    Return:
        [1] warped, N x C x H x W, warped image or feature map
    """
    n, c, h, w = flow.shape

    assert c == 1 or c == 2, f"invalid flow map dimension 1 or 2 ({c})!"

    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, device=image.device, dtype=image.dtype),
        torch.arange(w, device=image.device, dtype=image.dtype),
        indexing="ij",
    )

    grid_x = grid_x.view([1, 1, h, w]) - flow[:, 0, :, :].view([n, 1, h, w])
    grid_x = grid_x.permute([0, 2, 3, 1])

    if c == 2:
        grid_y = grid_y.view([1, 1, h, w]) - flow[:, 1, :, :].view(
            [n, 1, h, w])
        grid_y = grid_y.permute([0, 2, 3, 1])
    else:
        grid_y = grid_y.view([1, h, w, 1]).repeat(n, 1, 1, 1)

    grid_x = 2.0 * grid_x / (w - 1.0) - 1.0
    grid_y = 2.0 * grid_y / (h - 1.0) - 1.0
    grid_map = torch.concatenate((grid_x, grid_y), dim=-1)

    warped = F.grid_sample(image,
                           grid_map,
                           mode="bilinear",
                           padding_mode="zeros",
                           align_corners=True)
    return warped


def write_pfm_file(pfm_file, image, scale=1):
    """
    Write a Numpy array to a PFM file.
    """
    fp = open(pfm_file, "wb")

    color = None

    if image.dtype.name != "float32":
        raise Exception("Image dtype must be float32.")

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif (
        len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1
    ):  # greyscale
        color = False
    else:
        raise Exception("Image must have H x W x 3, H x W x 1 or H x W dimensions.")

    fp.write(b"PF\n" if color else b"Pf\n")
    fp.write(b"%d %d\n" % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == "<" or endian == "=" and sys.byteorder == "little":
        scale = -scale

    fp.write(b"%f\n" % scale)

    image.tofile(fp)


def load_camera_json(json_file):
    info = json.load(open(json_file, "r"))

    fx = info["fx"]
    baseline = info["baseline"]
    unit = info["unit"]
    return fx, baseline, unit


def main():
    FLAGS = gflags.FLAGS
    FLAGS(sys.argv)

    fx, baseline, unit = load_camera_json(\
        os.path.join(FLAGS.data_path, "camera.json"))

    l_img_files = glob.glob(\
        os.path.join(FLAGS.data_path, "image", "*_off_left_Img.png"))

    for l_img_file in tqdm(l_img_files, desc="loop over left image files.."):
        splits = os.path.basename(l_img_file).split("_")

        scene_name = "_".join(splits[:-4])

        uuid_tag = splits[-4]

        r_img_file = os.path.join(
            FLAGS.data_path, "image", f"{scene_name}_{uuid_tag}_off_right_Img.png")

        depth_file = glob.glob(\
            os.path.join(FLAGS.data_path, "image", f"{scene_name}_{uuid_tag}_on_*x*_depth_Img.png"))[0]

        l_img = cv.imread(l_img_file, cv.IMREAD_UNCHANGED)
        r_img = cv.imread(r_img_file, cv.IMREAD_UNCHANGED)

        h, w = l_img.shape[:2]

        depth = cv.imread(depth_file, cv.IMREAD_UNCHANGED).astype(np.float32)
        valid = np.where(depth < 1e-9)

        disp = fx * baseline / (depth * unit + 1e-15)
        disp[valid] = -1.0
        disp_file = os.path.join(\
            FLAGS.data_path, "disparity", f"{scene_name}_{uuid_tag}_on_{w}x{h}_disparity_Img.pfm")
        os.makedirs(os.path.dirname(disp_file), exist_ok=True)
        write_pfm_file(disp_file, np.flipud(disp), scale=1.0)

        l_img_warp = warp_by_flow_map(\
            image=torch.tensor(r_img.reshape([1, 1, h, w]), dtype=torch.float32),
            flow=torch.tensor(disp.reshape([1, 1, h, w]), dtype=torch.float32))
        l_img_warp = l_img_warp.detach().cpu().squeeze().numpy().astype(np.uint8)
        l_img_warp[valid] = 0.0

        l_img_warp_file = os.path.join(FLAGS.data_path, "warp",
                                       f"{scene_name}_{uuid_tag}_warp_Img.png")
        os.makedirs(os.path.dirname(l_img_warp_file), exist_ok=True)
        cv.imwrite(l_img_warp_file, l_img_warp)


if __name__ == "__main__":
    main()