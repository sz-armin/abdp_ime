from typing import Any, Callable, Generator, Optional, Sequence, Union
from abc import ABC, abstractmethod

class InAgentBase(ABC):
    """Abstract class for agents handling input and the input policy."""

    @abstractmethod
    def get(self, batch: Sequence[Any]) -> tuple[int, Sequence[Any]]:
        """Get a batch of inputs at asingle time step.

        Args:
            batch_size: The number of inputs to get.

        Returns:
            The time step and the batch of inputs.
        """
        pass