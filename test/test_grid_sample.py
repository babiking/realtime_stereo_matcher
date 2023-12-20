import os
import time
import torch
import onnxruntime
import numpy as np
from profiler.profiler import get_model_capacity
from cost_sample.grid_sample import TorchGridSampleSearch, TorchGridSampleParse

torch.manual_seed(23)


def export_cost_sample_to_onnx(model, onnx_file, n=1, c=1, h=32, w=64):
    cost_volume = torch.rand((n, c, h, w, w), dtype=torch.float32)
    flow_map = 2.0 * torch.rand((n, h, w, 1), dtype=torch.float32) - 1.0

    os.makedirs(os.path.dirname(onnx_file), exist_ok=True)

    torch.onnx.export(
        model,
        (cost_volume, flow_map),
        onnx_file,
        verbose=False,
        input_names=["cost_volume", "flow_map"],
        output_names=["output"],
        opset_version=16,
    )
    return cost_volume, flow_map


def run_cost_sample_onnx_inference(cost_volume, flow_map, onnx_file):
    session = onnxruntime.InferenceSession(onnx_file)

    start = time.time()
    sample_cost_volume = session.run(
        None,
        {
            "cost_volume": cost_volume.astype(np.float32),
            "flow_map": flow_map.astype(np.float32),
        },
    )
    end = time.time()
    elapsed = round(float(end - start), 6)
    print("{:<30}  onnx inference {:<8}seconds".format(onnx_file, elapsed))
    return sample_cost_volume


def test_cost_sample_model_capacity():
    search_range = 4

    n = 1
    c = 2 * search_range + 1
    h = 120
    w = 160
    d = 160

    search_inputs = (
        torch.rand(size=(n, 1, h, w, d), dtype=torch.float32),
        2.0 * torch.rand(size=(n, h, w, 1), dtype=torch.float32) - 1.0,
    )
    get_model_capacity(
        TorchGridSampleSearch(search_range), inputs=search_inputs, verbose=True
    )

    parse_inputs = (
        torch.rand(size=(n, c, h, w, d), dtype=torch.float32),
        2.0 * torch.rand(size=(n, h, w, 1), dtype=torch.float32) - 1.0,
    )
    get_model_capacity(TorchGridSampleParse(), inputs=parse_inputs, verbose=True)


def test_cost_sample_onnx():
    search_onnx_file = "onnx/grid_sample_search.onnx"
    search_range = 4
    cost_volume_search, flow_map_search = export_cost_sample_to_onnx(
        model=TorchGridSampleSearch(search_range),
        onnx_file=search_onnx_file,
        n=1,
        c=1,
        h=120,
        w=160,
    )
    run_cost_sample_onnx_inference(
        cost_volume_search.cpu().detach().numpy(),
        flow_map_search.cpu().detach().numpy(),
        search_onnx_file,
    )

    parse_onnx_file = "onnx/grid_sample_parse.onnx"
    cost_volume_parse, flow_map_parse = export_cost_sample_to_onnx(
        model=TorchGridSampleParse(),
        onnx_file=parse_onnx_file,
        n=1,
        c=2 * search_range + 1,
        h=120,
        w=160,
    )
    run_cost_sample_onnx_inference(
        cost_volume_parse.cpu().detach().numpy(),
        flow_map_parse.cpu().detach().numpy(),
        parse_onnx_file,
    )


def main():
    test_cost_sample_model_capacity()
    test_cost_sample_onnx()


if __name__ == "__main__":
    main()
