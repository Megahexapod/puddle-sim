"""First unit tests for puddle_sim — the Liebig weighted geometric mean (_geo).

Run them from the project root with:  pytest
(install pytest first:  pip install -r requirements-dev.txt)

_geo is the heart of how creatures score their habitat: several factors
(shade, water quality, food, ...) combine so that ANY single factor hitting
zero drags the whole score to zero. These tests guard exactly that behaviour.

The habit to learn here: don't test "one example and eyeball the number". Test
the *properties* the function must ALWAYS have. Each test below names one
property in its function name — that's not decoration, it's the point.
"""
import numpy as np
from pytest import approx

from puddle_sim.creatures import _geo


def arr(x: float) -> np.ndarray:
    """_geo runs on numpy arrays (one score per map cell). For a unit test we
    just wrap a single number as a 1-element array and read result[0]."""
    return np.array([x], dtype=np.float32)


def test_all_factors_one_gives_one():
    # Property: every factor perfect (1.0) -> perfect score.
    out = _geo([(arr(1.0), 1.0), (arr(1.0), 1.0)])
    assert out[0] == approx(1.0)


def test_any_factor_zero_kills_the_score():
    # Property (the whole reason _geo exists): one factor at zero -> score zero,
    # however good the others are. This is Liebig's law of the minimum.
    out = _geo([(arr(1.0), 1.0), (arr(0.0), 1.0)])
    assert out[0] == approx(0.0)


# ---- 換你寫 -------------------------------------------------------------------
# 照上面兩個的 pattern。先想清楚「這函式一定成立的性質是什麼」，test 的名字
# 就寫那個性質，body 用一個能證明它的例子。寫完跑 pytest 看全綠，丟給我 review。
# 注意：浮點數不要用 ==，用 approx（例：assert out[0] == approx(0.5)）。

def test_all_factors_equal_returns_that_value():
    # Property: 每個 factor 都等於 c（不管權重多少）-> 結果就是 c。
    # 提示：c=0.5，放兩個不同權重的 factor（例如權重 2 和權重 5），結果仍該是 0.5。
    out = _geo([(arr(0.5), 2.0), (arr(0.5), 5.0)])
    assert out[0] == approx(0.5)

def test_factors_are_clipped_to_0_1():
    # Property: factor 超過 1 當作 1、小於 0 當作 0（_geo 內部有 np.clip）。
    # 提示：_geo([(arr(5.0), 1.0)])[0] 該是 1.0；arr(-3.0) 那個該是 0.0。兩個 assert。
    out_high = _geo([(arr(5.0), 1.0)])
    assert out_high[0] == approx(1.0)
    out_low = _geo([(arr(-3.0), 1.0)])
    assert out_low[0] == approx(0.0)