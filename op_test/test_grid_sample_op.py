import pytest
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from op_test.base_op import BaseOp
from op_test.helper import TestHelper


class GridSampleOp(BaseOp):
    def get_random_inputs(self, n=1, c=32, h=30, w=40):
        input = torch.randn(size=(n, c, h, w), dtype=torch.float32)

        grid = torch.rand(size=(n, h, w, 2), dtype=torch.float32)
        grid = 2.0 * grid - 1.0
        return (input, grid)

    def get_output_number(self):
        return 1

    def forward(self, input, grid):
        sample = F.grid_sample(
            input=input,
            grid=grid,
            padding_mode="zeros",
            mode="bilinear",
            align_corners=True,
        )
        return [sample]


def test_grid_sample_op():
    test_helper = TestHelper(
        model=GridSampleOp(),
        dump_path=os.path.join(os.path.dirname(__file__), "grid_sample_test"),
    )
    test_helper.execute(n=20)


if __name__ == "__main__":
    test_grid_sample_op()
