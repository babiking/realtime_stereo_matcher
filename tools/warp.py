import torch
import torch.nn.functional as F


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
        grid_y = grid_y.view([1, 1, h, w]) - flow[:, 1, :, :].view([n, 1, h, w])
        grid_y = grid_y.permute([0, 2, 3, 1])
    else:
        grid_y = grid_y.view([1, h, w, 1]).repeat(n, 1, 1, 1)

    grid_x = 2.0 * grid_x / (w - 1.0) - 1.0
    grid_y = 2.0 * grid_y / (h - 1.0) - 1.0
    grid_map = torch.concatenate((grid_x, grid_y), dim=-1)

    warped = F.grid_sample(
        image, grid_map, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    return warped
