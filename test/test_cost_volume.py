from profiler.profiler import get_random_sample_data, get_model_capacity
from tools.convert import export_stereo_model_to_onnx
from cost_volume.interweave import TorchInterweaveCost
from cost_volume.inner_product import TorchInnerProductCost
from cost_volume.concatenate import TorchConcatenateCost
from cost_volume.groupwise import TorchGroupwiseCost


def test_cost_volume_model_capacity():
    left = get_random_sample_data(n=1, c=128, h=120, w=160)
    right = get_random_sample_data(n=1, c=128, h=120, w=160)

    get_model_capacity(
        module=TorchInterweaveCost(),
        inputs=(left, right),
        verbose=True,
    )

    get_model_capacity(
        module=TorchInnerProductCost(),
        inputs=(left, right),
        verbose=True,
    )

    get_model_capacity(
        module=TorchConcatenateCost(max_disparity=128),
        inputs=(left, right),
        verbose=True,
    )

    get_model_capacity(
        module=TorchGroupwiseCost(n_groups=8, max_disparity=128),
        inputs=(left, right),
        verbose=True,
    )


def test_cost_volume_onnx():
    export_stereo_model_to_onnx(
        model=TorchInterweaveCost(),
        onnx_file="onnx/interweave_cost.onnx",
    )

    export_stereo_model_to_onnx(
        model=TorchInnerProductCost(),
        onnx_file="onnx/product_cost.onnx",
    )

    export_stereo_model_to_onnx(
        model=TorchConcatenateCost(max_disparity=16),
        onnx_file="onnx/concatenate_cost.onnx",
    )

    export_stereo_model_to_onnx(
        model=TorchGroupwiseCost(n_groups=4, max_disparity=16),
        onnx_file="onnx/groupwise_cost.onnx",
    )


def main():
    test_cost_volume_model_capacity()
    test_cost_volume_onnx()


if __name__ == "__main__":
    main()
