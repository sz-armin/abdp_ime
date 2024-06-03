import pytest

from preprocess.mec2alignment import AlignmentExtractor


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ((["あ", "い", "う"], ["a", "i", "u"]), "1 2 3 9999 9999"),
        ((["かんばつ"], ["間伐"]), "4 9999 9999"),
        (
            (
                [
                    "にちじ",
                    "いち",
                    "れい",
                    "がつ",
                    "いち",
                    "はち",
                    "にち",
                    "（",
                    "ど",
                    "）",
                    "いち",
                    "よ",
                    "じ",
                    "〜",
                    "いち",
                    "ご",
                    "じ",
                ],
                [
                    "日時",
                    "一",
                    "零",
                    "月",
                    "一",
                    "八",
                    "日",
                    "（",
                    "土",
                    "）",
                    "一",
                    "四",
                    "時,",
                    "〜",
                    "一",
                    "五",
                    "時",
                ],
            ),
            "3 5 7 9 11 13 15 16 17 18 20 21 22 23 25 26 27 9999 9999",
        ),
    ],
)
def test_align_from_sample(test_input, expected):
    exctractor = AlignmentExtractor()
    alignment = exctractor.align_from_sample(test_input)
    assert alignment == expected


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ((["あ", "い", "う"], []), "1 2 3 9999 9999"),
    ],
)
def test_align_from_sample_assertions(test_input, expected):
    with pytest.raises(AssertionError):
        exctractor = AlignmentExtractor()
        exctractor.align_from_sample(test_input)
