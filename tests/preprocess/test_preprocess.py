import pytest

from preprocess.parse_prep_bccwj import has_uncommon_chars


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ("゠ァアィイゥウェエォオ", False),
        ("あいうえおかきくけこ", False),
        ("？。〖〶 〷 〸", False),
        ("ＡＢＣＤＥ！＂＃＄％＆", False),
        ("ABCDabcd.,?-=+)(*&^%$#@!~", False),
        ("ABCDabcd.,?-=+)(*&^%$#@!~", False),
        ("日曜改定", True),
        ("日曜改定ォオえ お.", True),
        ("★■", True),
        ("三", True),
    ],
)
def test_has_uncommon_chars(test_input, expected):
    assert has_uncommon_chars(test_input) == expected
