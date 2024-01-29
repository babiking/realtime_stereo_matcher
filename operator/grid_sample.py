import os
import onnxruntime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class GridSampleOperator(nn.Module):
    def forward(self, input: torch.Tensor, grid: torch.Tensor):
        return F.grid_sample(
            input=input,
            grid=grid,
            padding_mode="zeros",
            mode="bilinear",
            align_corners=True,
        )


def get_random_inputs(n=1, c=64, h=120, w=160):
    input = np.random.randn(n, c, h, w).astype(np.float32)

    grid = np.random.random(size=[n, h, w, 2]).astype(np.float32)
    grid = 2.0 * grid - 1.0
    return input, grid


def export_to_onnx(onnx_file):
    os.makedirs(os.path.dirname(onnx_file), exist_ok=True)

    model = GridSampleOperator()

    input, grid = [torch.tensor(x, dtype=torch.float32) for x in get_random_inputs()]

    torch.onnx.export(
        model,
        (input, grid),
        onnx_file,
        verbose=False,
        input_names=["input", "grid"],
        output_names=["sample"],
        opset_version=16,
    )
    return model


def run_onnx_inference(input, grid, onnx_file):
    session = onnxruntime.InferenceSession(onnx_file)

    sample = session.run(
        None,
        {"input": input.astype(np.float32), "grid": grid.astype(np.float32)},
    )
    return sample[0]


def run_n_compares(n=20):
    save_path = os.path.join(os.path.dirname(__file__), "grid_sample_op")

    onnx_file = os.path.join(save_path, "grid_sample.onnx")
    model = export_to_onnx(onnx_file)

    for i in tqdm(range(n), desc="loop over GridSample randoms..."):
        input, grid = get_random_inputs()

        torch_sample = model(
            input=torch.tensor(input, dtype=torch.float32),
            grid=torch.tensor(grid, dtype=torch.float32),
        )
        torch_sample = torch_sample.cpu().detach().numpy()

        onnx_sample = run_onnx_inference(input, grid, onnx_file)

        assert np.allclose(
            torch_sample, onnx_sample, rtol=1e-6, atol=1e-6
        ), f"ITER-{i} torch and onnx mismatch!"

        np.save(os.path.join(save_path, f"{i:04d}_input.npy"), input)
        np.save(os.path.join(save_path, f"{i:04d}_grid.npy"), grid)
        np.save(os.path.join(save_path, f"{i:04d}_sample.npy"), torch_sample)


if __name__ == "__main__":
    run_n_compares(n=20)
