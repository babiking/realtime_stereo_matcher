import pytest
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from op_test.base_op import BaseOp
from op_test.helper import TestHelper


class SliceOp(BaseOp):
    def __init__(self, repeat=24, reverse=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.repeat = repeat
        self.reverse = reverse

    def get_random_inputs(self, n=1, c=32, h=30, w=40):
        input0 = torch.randn(size=(n, c, h, w), dtype=torch.float32)
        return (input0,)

    def get_output_number(self):
        return 1

    def forward(self, input0):
        n, c, h, w = input0.shape

        output0 = torch.zeros(
            [n, c, self.repeat, h, w], dtype=input0.dtype, device=input0.device
        )

        for i in range(self.repeat):
            if i == 0:
                output0[:, :, 0, :, :] = input0
            else:
                output0[:, :, i, :, i:] = (
                    input0[:, :, :, i:] if not self.reverse else input0[:, :, :, :-i]
                )
        return [output0]


def test_slice_op():
    test_helper = TestHelper(
        model=SliceOp(repeat=24, reverse=True),
        dump_path=os.path.join(os.path.dirname(__file__), "slice_test"),
    )
    test_helper.execute(n=20)


if __name__ == "__main__":
    test_slice_op()
