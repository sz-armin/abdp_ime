from typing import Any, Generator, Sequence, Union

from in_agent.base import InAgentBase


class AllInAgent(InAgentBase):
    def __init__(self) -> None:
        pass

    def get(
        self, batch: Sequence[Any], inc_align: bool = True
    ) -> Generator[Union[Sequence[Any], tuple[Sequence[Any], ...]], None, None]:
        src = list(map(int, batch[0][0].split()))
        align = list(map(int, batch[0][1].split()))
        if not inc_align:
            yield [src]
        else:
            yield [src], [align]
