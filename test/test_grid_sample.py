import torch
from profiler.profiler import get_model_capacity
from tools.convert import export_stereo_model_to_onnx
from cost_sample.grid_sample import TorchGridSampleSearch, TorchGridSampleParse


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


def main():
    test_cost_sample_model_capacity()


if __name__ == "__main__":
    main()
