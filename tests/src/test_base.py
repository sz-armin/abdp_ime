from src.models.base import PositionalEncoding
import torch
import pytest


def test_positional_encoding():
    positional_encoding = PositionalEncoding(d_model=2, dropout=0)

    x = torch.zeros(1, 3, 2)
    output = positional_encoding(x)
    assert output.shape == (1, 3, 2)

    expected_output = torch.tensor(
        [
            [   
                [0.0000, 1.0000],
                [0.8415, 0.5403],
                [0.9093, -0.4161],
            ]
        ]
    )
    assert torch.allclose(output, expected_output, rtol=0.001)

    x = torch.zeros(1, 1, 2)
    output = positional_encoding(x, fixed_pos=2)
    expected_output = torch.tensor([[[0.8415, 0.5403]]])
    assert torch.allclose(output, expected_output, rtol=0.001)

    positional_encoding = PositionalEncoding(d_model=128, dropout=0)
    x = torch.zeros(8, 25, 128)
    output = positional_encoding(x)
    output = (output.mul(output)).sum(dim=-1)
    assert torch.allclose(output, torch.ones(8, 25) * 64, rtol=0.001)


