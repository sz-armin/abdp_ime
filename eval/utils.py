import math
from typing import Sequence


def get_chunk(seq: Sequence, nth, number_of_chunks):
    """
    Split a sequence into `number_of_chunks` chunks and return the `n`th chunk.
    """
    chunk_size = math.ceil(len(seq) / number_of_chunks)

    if nth < 0:
        raise ValueError("nth must be non-negative")
    elif nth >= number_of_chunks:
        raise ValueError("nth must be less than number_of_chunks")

    return seq[chunk_size * nth : chunk_size * (nth + 1)]
