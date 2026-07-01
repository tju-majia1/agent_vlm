"""
余弦相似度单元测试 —— 防止退回成"未归一化的点积"。

跑法：
    pytest tests/test_retriever_cosine.py
"""

from __future__ import annotations

import math

from mobilerun.skills.retriever import cosine


def test_identical_direction_is_one_regardless_of_magnitude():
    # 方向相同但模长差 10 倍：真正的 cosine 应为 1.0，点积会得到 10。
    a = [1.0, 0.0, 0.0]
    b = [10.0, 0.0, 0.0]
    assert math.isclose(cosine(a, b), 1.0, rel_tol=1e-9)


def test_orthogonal_is_zero():
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_opposite_is_minus_one():
    assert math.isclose(cosine([1.0, 2.0], [-1.0, -2.0]), -1.0, rel_tol=1e-9)


def test_zero_vector_is_zero():
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_dim_mismatch_is_zero():
    assert cosine([1.0, 2.0], [1.0]) == 0.0
