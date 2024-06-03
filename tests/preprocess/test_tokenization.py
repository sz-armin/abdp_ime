from preprocess.tokenization import fix_sample_bpe_alignemnt
import pytest


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (
            ("<s> プロ デ ュー サー</w> の</w> メ チ エ</w> </s>", "7 8 11 9999 9999"),
            "7 7 7 7 8 11 11 11 9999 9999",
        ),
        (
            (
                "<s> 「</w> いよいよ</w> 後半</w> 戦</w> です</w> 。</w> </s>",
                "1 5 9 11 13 14 9999 9999",
            ),
            "1 5 9 11 13 14 9999 9999",
        ),
    ],
)
def test_fix_sample_bpe_alignemnt(test_input, expected):
    assert fix_sample_bpe_alignemnt(*test_input) == expected
