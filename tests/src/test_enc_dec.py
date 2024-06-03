from src.models.enc_dec import TransformerEncDec
import torch
import pytest
import math

# TODO
@pytest.mark.parametrize(
    "test_input,expected",
    [
        (
            (
                torch.tensor([[2, 9999, 9999]]),
                torch.tensor([[1, 84, 122, 2]]),
            ),
            torch.tensor([[False, False, False, True]]),
        ),
        (
            (
                torch.tensor([[1, 3, 9999, 9999]]),
                torch.tensor([[1, 84, 122, 50, 2]]),
            ),
            torch.tensor([[False, False, True, False, True]]),
        ),
    ],
)
def test_salign_to_balign(test_input, expected):
    model = TransformerEncDec(500, 500)

    assert (model.salign_to_balign(*test_input) == expected).all()


@pytest.mark.parametrize(
    "current_step, expected",
    [
        (3, 3 / 5),
        (5, (math.cos(0 / (math.pi * 5)) + 1) / 2),
        (8, (math.cos(30 / (math.pi * 5)) + 1) / 2),
        (10, 0),
        (11, 0),
    ],
    ids=[
        "below warmup",
        "at warmup",
        "above warmup",
        "at max steps",
        "above max steps",
    ],
)
def test_rate_cosine(current_step, expected):
    model = TransformerEncDec(500, 500)
    model.warmup = 5
    model.max_steps = 10

    assert math.isclose(model.rate_cosine(current_step), expected, abs_tol=0.001)


@pytest.mark.parametrize(
    "current_step, expected",
    [
        (0, 0.01 * (256 ** (-0.5)) * (1 ** (-0.5))),
        (1, 0.01 * (256 ** (-0.5)) * (1 ** (-0.5))),
        (3, 0.01 * (256 ** (-0.5)) * (3 ** (-0.5))),
        (5, 0.01 * (256 ** (-0.5)) * (5 ** (-1.5))),
        (8, 0.01 * (256 ** (-0.5)) * (5 ** (-1.5))),
        (10, 0.01 * (256 ** (-0.5)) * (5 ** (-1.5))),
    ],
    ids=[
        "current_step=0",
        "current_step=1",
        "current_step<warmup",
        "current_step=warmup",
        "current_step>warmup",
        "current_step>warmup",
    ],
)
def test_rate_trans(current_step, expected):
    model = TransformerEncDec(500, 500)
    model.warmup = 5
    model.d_model = 256
    model.lr_factor = 0.01

    assert math.isclose(model.rate_trans(current_step), expected, abs_tol=0.001)
