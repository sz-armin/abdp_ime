from typing import Any, Generator, Sequence, Union

from in_agent.base import InAgentBase


class CharSuffixInAgent(InAgentBase):
    def __init__(self, inc_eos=None) -> None:
        self.inc_eos = inc_eos

    def get(
        self,
        batch: Sequence[Any],
        inc_align: bool = True,
        inc_eos: bool = False,  # TODO inc_eos
    ) -> Generator[Union[Sequence[Any], tuple[Sequence[Any], ...]], None, None]:
        if self.inc_eos is not None:
            inc_eos = self.inc_eos
        src = list(map(int, batch[0][0].split()))
        align = list(map(int, batch[0][1].split()))
        if not inc_align:
            for i in range(1, len(src) + 1):
                src_prefix = src[:i]
                if src_prefix[-1] != 2 and inc_eos:
                    src_prefix.append(2)
                yield [src_prefix]
        else:
            for i in range(1, len(src) + 1):
                src_prefix = src[:i]
                align_prefix = align[:i]
                if src_prefix[-1] != 2 and inc_eos:
                    src_prefix.append(2)
                    align_prefix.append(align_prefix[-1])
                yield [src_prefix], [align_prefix]


class CharInAgent(InAgentBase):
    def __init__(self) -> None:
        pass

    def get(
        self, batch: Sequence[Any], inc_align: bool = True
    ) -> Generator[Union[Sequence[Any], tuple[Sequence[Any], ...]], None, None]:
        src = list(map(int, batch[0][0].split()))
        align = list(map(int, batch[0][1].split()))
        if not inc_align:
            for s in src:
                yield [s]
        else:
            for s in src:
                yield [s], [align]
