from abc import ABC, abstractmethod
from typing import Any, Sequence, Union


class OutAgentBase(ABC):
    """Abstract class for agents handling output and the output policy."""

    @abstractmethod
    def put(self, batch: Union[int, Sequence[Any]]) -> None:
        """Put a batch of outputs.

        Args:
            batch: The batch of inputs.
        """
        pass