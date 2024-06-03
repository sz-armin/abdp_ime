import numpy as np
import pytest

from utils.score import Scorer


def test_prf():
    scorer = Scorer()
    assert np.array_equal(scorer._lcs("あいう", "あいう"), np.array([1, 1, 1]))
    assert np.array_equal(scorer._lcs("あいう", "かきく"), np.array([0, 0, 0]))
    assert np.array_equal(scorer._lcs("あい", "あ"), np.array([0.5, 1, 1 / 1.5]))
    assert np.array_equal(scorer._lcs("あ", "あい"), np.array([1, 0.5, 1 / 1.5]))
    assert np.array_equal(
        scorer._lcs("あいう", "かいく"), np.array([1 / 3, 1 / 3, (2 / 9) / (2 / 3)])
    )

    with pytest.raises(ZeroDivisionError):
        scorer._lcs("", "あいう")

    with pytest.raises(ZeroDivisionError):
        scorer._lcs("あいう", "")


def test_cer():
    scorer = Scorer()
    assert scorer._cer("あいう", "あいう") == 0 / 3.0
    assert scorer._cer("あいう", "かきく") == 3 / 3.0
    assert scorer._cer("あい", "あ") == 1 / 1.0
    assert scorer._cer("あ", "あい") == 1 / 2
    assert scorer._cer("あいう", "かいく") == 2 / 3.0

    assert scorer._cer("", "あいう") == 3 / 3

    with pytest.raises(ZeroDivisionError):
        scorer._lcs("あいう", "")


@pytest.mark.integtest
def test_score_paralell():
    scorer = Scorer()

    src_list = ["あいう", "あいう", "あいう", "あいう"]
    tgt_list = ["あいう", "あいう", "あいう", "あいう"]
    assert scorer.score_paralell(src_list, tgt_list) == {
        "P": 1,
        "R": 1,
        "F2": 1,
        "Acc": 1,
        "CER": 0,
        "TgtLen": 3,
        "SrcLen": 3,
        "Count": 4,
    }

    src_list = ["あい", "あい", "あい"]
    tgt_list = ["か", "か", "かく"]
    assert scorer.score_paralell(src_list, tgt_list) == {
        "P": 0,
        "R": 0,
        "F2": 0,
        "Acc": 0,
        "CER": 5 / 3,
        "TgtLen": 4 / 3,
        "SrcLen": 2,
        "Count": 3,
    }

    src_list = ["あい", "あ"]
    tgt_list = ["あ", "あい"]
    assert scorer.score_paralell(src_list, tgt_list) == {
        "P": 0.75,
        "R": 0.75,
        "F2": 1 / 1.5,
        "Acc": 0,
        "CER": 1.5 / 2,
        "TgtLen": 1.5,
        "SrcLen": 1.5,
        "Count": 2,
    }
