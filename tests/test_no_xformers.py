import torch

from mdm.model.dinov2_rgbd.layers.attention import MemEffAttention
from mdm.model.dinov2_rgbd.layers.block import Block, NestedTensorBlock


def test_nested_tensor_block_falls_back_to_native_attention(monkeypatch):
    monkeypatch.setattr("mdm.model.dinov2_rgbd.layers.block.XFORMERS_AVAILABLE", False)
    monkeypatch.setattr("mdm.model.dinov2_rgbd.layers.attention.XFORMERS_AVAILABLE", False)
    block = NestedTensorBlock(dim=8, num_heads=2, attn_class=MemEffAttention).eval()
    inputs = [torch.randn(1, 5, 8), torch.randn(1, 7, 8)]

    expected = [Block.forward(block, value.clone()) for value in inputs]
    actual = block([value.clone() for value in inputs])

    assert len(actual) == len(inputs)
    for actual_value, expected_value in zip(actual, expected):
        torch.testing.assert_close(actual_value, expected_value)
