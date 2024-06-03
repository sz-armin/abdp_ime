import pytest

from eval.utils import *


@pytest.mark.parametrize(
    "seq, nth, number_of_chunks, expected_chunk",
    [
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 2, 3, [9, 10]),
        (["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"], 0, 4, ["a", "b", "c"]),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1, 4, [4, 5, 6]),
    ],
)
def test_get_chunk(
    seq: Sequence, nth: int, number_of_chunks: int, expected_chunk: Sequence
):
    # Ensure that the function is correct
    assert get_chunk(seq, nth, number_of_chunks) == expected_chunk

    # Ensure that the function raises a ValueError if nth is negative
    with pytest.raises(ValueError):
        get_chunk(seq, -1, number_of_chunks)

    # Ensure that the function raises a ValueError if nth is greater than or equal to number_of_chunks
    with pytest.raises(ValueError):
        get_chunk(seq, number_of_chunks, number_of_chunks)
        get_chunk(seq, number_of_chunks + 1, number_of_chunks)
