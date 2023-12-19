import os
import torch


def export_stereo_model_to_onnx(
    model,
    onnx_file,
    n=1,
    c=8,
    h=32,
    w=64,
):
    l_fmap = torch.rand((n, c, h, w), dtype=torch.float32)
    r_fmap = torch.rand((n, c, h, w), dtype=torch.float32)

    os.makedirs(os.path.dirname(onnx_file), exist_ok=True)

    torch.onnx.export(
        model,
        (l_fmap, r_fmap),
        onnx_file,
        verbose=False,
        input_names=["left", "right"],
        output_names=["output"],
        opset_version=16,
    )
