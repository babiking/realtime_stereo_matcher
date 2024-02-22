import pytest
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from op_test.base_op import BaseOp
from op_test.helper import TestHelper


class MyEinsumOp(BaseOp):
    def get_random_inputs(self, n=1, c=32, h=30, w=40):
        input0 = torch.randn(size=(n, c, h, w), dtype=torch.float32)
        input1 = torch.randn(size=(n, c, h, w), dtype=torch.float32)
        return (input0, input1)

    def get_output_number(self):
        return 1

    def forward(self, input0, input1):
        naive_output = torch.einsum("aijk,aijh->ajkh", input0, input1)
        return [naive_output]


def test_my_einsum_op():
    test_helper = TestHelper(
        model=MyEinsumOp(),
        dump_path=os.path.join(os.path.dirname(__file__), "my_einsum_test"),
    )
    test_helper.execute(n=20)


if __name__ == "__main__":
    test_my_einsum_op()
