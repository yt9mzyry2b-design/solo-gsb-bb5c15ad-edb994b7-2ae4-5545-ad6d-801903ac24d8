"""纸质乐谱小册子拼版核心的单元测试。"""

from __future__ import annotations

import itertools
import random

import pytest

from app.booklet import (
    MAX_PAGES,
    BookletError,
    plan_booklet,
)


def _brute_force(ends, max_sheets):
    """暴力枚举全部合法分帖，返回最优 (总空白, 帖数, 各帖末页序列)。

    目标依次为：总空白最少 → 帖数最少 → 末页序列字典序最小。
    无任何合法分帖时返回 None。
    """
    m = len(ends)
    best = None
    # 枚举 m-1 个段间断点的全部子集：每个子集对应一种分帖
    for mask in range(1 << (m - 1)):
        groups = []
        start = 0
        for i in range(m - 1):
            if mask >> i & 1:
                groups.append((start, i))
                start = i + 1
        groups.append((start, m - 1))
        total_blanks = 0
        seq = []
        feasible = True
        for a, b in groups:
            length = ends[b] - (ends[a - 1] if a > 0 else 0)
            if (length + 3) // 4 > max_sheets:
                feasible = False
                break
            total_blanks += (-length) % 4
            seq.append(ends[b])
        if not feasible:
            continue
        key = (total_blanks, len(groups), tuple(seq))
        if best is None or key < best:
            best = key
    return best


def _check_layout(result):
    """校验拼版页序的结构不变量：

    - 每帖 4·印张数 个页位 = 原谱页数 + 空白页数；
    - 各帖所有印张上的非空页码恰好是该帖原谱页范围的一个排列；
    - 空白数与 null 个数一致；
    - 同一印张正面/背面两页页位下标之和恒为 4S-1（页码和恒为
      2·首页 + 4S - 1），即骑马订"末页配首页、次页配倒数第二页"。
    """
    for sig in result.signatures:
        length = sig.end_page - sig.start_page + 1
        assert 4 * sig.sheet_count == length + sig.blank_pages
        assert len(sig.sheets) == sig.sheet_count
        assert [s.sheet_index for s in sig.sheets] == list(
            range(1, sig.sheet_count + 1)
        )
        pages = []
        pair_sum = 2 * sig.start_page + 4 * sig.sheet_count - 1
        for sheet in sig.sheets:
            for side in (sheet.front, sheet.back):
                for value in (side.left, side.right):
                    if value is not None:
                        pages.append(value)
                if side.left is not None and side.right is not None:
                    assert side.left + side.right == pair_sum
        assert sorted(pages) == list(range(sig.start_page, sig.end_page + 1))
        nulls = 4 * sig.sheet_count - len(pages)
        assert nulls == sig.blank_pages
    assert result.signature_count == len(result.signatures)
    assert result.total_blank_pages == sum(
        s.blank_pages for s in result.signatures
    )
    # 各帖首尾相接、完整覆盖 1..page_count
    assert result.signatures[0].start_page == 1
    assert result.signatures[-1].end_page == result.page_count
    for left, right in zip(result.signatures, result.signatures[1:]):
        assert right.start_page == left.end_page + 1


def _check_against_brute_force(page_count, ends, max_sheets):
    result = plan_booklet(page_count, ends, max_sheets)
    best = _brute_force(ends, max_sheets)
    assert best is not None
    assert (
        result.total_blank_pages,
        result.signature_count,
        tuple(s.end_page for s in result.signatures),
    ) == best
    _check_layout(result)
    return result


# ---------------------------------------------------------------- 确定性用例


def test_single_signature_exact_multiple_of_four():
    result = plan_booklet(8, [8], 2)
    assert result.signature_count == 1
    assert result.total_blank_pages == 0
    (sig,) = result.signatures
    assert (sig.signature_index, sig.start_page, sig.end_page) == (1, 1, 8)
    assert sig.blank_pages == 0
    assert sig.sheet_count == 2
    # 骑马订：外层正面 末页/首页、背面 次页/倒数第二页，向内交替
    assert [(s.front.left, s.front.right) for s in sig.sheets] == [(8, 1), (6, 3)]
    assert [(s.back.left, s.back.right) for s in sig.sheets] == [(2, 7), (4, 5)]
    _check_layout(result)


def test_blank_pages_padded_at_signature_tail_as_null():
    result = plan_booklet(6, [6], 2)
    (sig,) = result.signatures
    assert sig.blank_pages == 2
    assert sig.sheet_count == 2
    # 页位 [1,2,3,4,5,6,None,None]：空白只出现在帖尾
    assert [(s.front.left, s.front.right) for s in sig.sheets] == [
        (None, 1),
        (6, 3),
    ]
    assert [(s.back.left, s.back.right) for s in sig.sheets] == [
        (2, None),
        (4, 5),
    ]
    _check_layout(result)


def test_single_sheet_signature():
    result = plan_booklet(3, [3], 1)
    (sig,) = result.signatures
    assert sig.sheet_count == 1
    assert sig.blank_pages == 1
    sheet = sig.sheets[0]
    assert (sheet.front.left, sheet.front.right) == (None, 1)
    assert (sheet.back.left, sheet.back.right) == (2, 3)
    _check_layout(result)


def test_minimize_total_blanks_first():
    # 两段各 2 页、每帖至多 1 张印张（4 页）：合成一帖 0 空白，
    # 优于分两帖各补 2 页空白（共 4 页）。
    result = plan_booklet(4, [2, 4], 1)
    assert result.signature_count == 1
    assert result.total_blank_pages == 0
    assert result.signatures[0].start_page == 1
    assert result.signatures[0].end_page == 4
    _check_layout(result)


def test_minimize_signature_count_on_blank_tie():
    # 两段各 3 页、每帖至多 2 张印张：合成一帖（6 页补 2 空白）与
    # 分两帖（各补 1 空白）总空白同为 2，取帖数更少的一帖方案。
    result = plan_booklet(6, [3, 6], 2)
    assert result.total_blank_pages == 2
    assert result.signature_count == 1
    assert result.signatures[0].end_page == 6
    _check_layout(result)


def test_lexicographically_smallest_end_pages_on_full_tie():
    # 三段各 2 页、每帖至多 1 张印张：[1..2][3..6] 与 [1..4][5..6]
    # 总空白同为 2、帖数同为 2，取末页序列字典序最小的 [2, 6]。
    result = plan_booklet(6, [2, 4, 6], 1)
    assert result.total_blank_pages == 2
    assert result.signature_count == 2
    assert [s.end_page for s in result.signatures] == [2, 6]
    first, second = result.signatures
    assert (first.start_page, first.end_page, first.blank_pages) == (1, 2, 2)
    assert (second.start_page, second.end_page, second.blank_pages) == (3, 6, 0)
    _check_layout(result)


def test_each_section_own_signature_when_merging_costs_blanks():
    # 两段各 4 页、每帖至多 2 张印张：各自独立成帖 0 空白，
    # 优于合成一帖（8 页也 0 空白但……同为 0 空白时帖数少者胜）。
    result = plan_booklet(8, [4, 8], 1)
    # 每帖至多 1 张印张（4 页），只能两段分两帖
    assert result.signature_count == 2
    assert result.total_blank_pages == 0
    assert [s.end_page for s in result.signatures] == [4, 8]
    _check_layout(result)


def test_merging_beats_splitting_when_blanks_tie():
    # 每帖至多 2 张印张时上例可合成一帖：空白同为 0，取帖数少者。
    result = plan_booklet(8, [4, 8], 2)
    assert result.signature_count == 1
    assert result.total_blank_pages == 0
    _check_layout(result)


def test_many_sections_large_capacity_single_signature():
    result = plan_booklet(12, [2, 5, 7, 12], 3)
    assert result.signature_count == 1
    assert result.total_blank_pages == 0
    assert result.signatures[0].sheet_count == 3
    _check_layout(result)


def test_capacity_forces_split_even_with_more_blanks():
    # 三段各 4 页、每帖至多 2 张印张：整本 12 页需 3 张印张贴不下，
    # 只能分两帖；[1..4]+[5..12] 与 [1..8]+[9..12] 空白同为 0、帖数同
    # 为 2，取末页序列字典序最小的 [4, 12]。
    result = plan_booklet(12, [4, 8, 12], 2)
    assert result.signature_count == 2
    assert result.total_blank_pages == 0
    assert [s.end_page for s in result.signatures] == [4, 12]
    _check_layout(result)


# ---------------------------------------------------------------- 非法输入


def test_single_section_over_capacity_raises():
    with pytest.raises(BookletError, match="超过每帖最多"):
        plan_booklet(12, [12], 2)


def test_middle_section_over_capacity_raises():
    with pytest.raises(BookletError, match="第 2 段"):
        plan_booklet(12, [2, 7, 12], 1)


def test_capacity_boundary_exact_fit_accepted():
    # 单段 8 页恰需 2 张印张，每帖至多 2 张：合法
    result = plan_booklet(8, [8], 2)
    assert result.signature_count == 1
    with pytest.raises(BookletError, match="超过每帖最多"):
        plan_booklet(9, [9], 2)


@pytest.mark.parametrize(
    "ends,fragment",
    [
        ([], "不能为空"),
        ([4], "必须等于方案页数"),
        ([2, 4], "必须等于方案页数"),
        ([8, 4], "严格递增"),
        ([4, 4], "严格递增"),
        ([0, 4], "越界"),
        ([5], "必须等于方案页数"),
        ([2, 9], "越界"),
    ],
)
def test_invalid_section_end_pages_raise(ends, fragment):
    with pytest.raises(BookletError, match=fragment):
        plan_booklet(8, ends, 2)


@pytest.mark.parametrize("bad", [1.5, 2.0, True, False, "2", None])
def test_non_integer_end_pages_raise(bad):
    with pytest.raises(BookletError, match="必须为整数"):
        plan_booklet(8, [2, bad, 8], 2)


@pytest.mark.parametrize("bad", [0, -1, 1.5, 2.0, True, "2", None])
def test_invalid_max_sheets_raise(bad):
    with pytest.raises(BookletError):
        plan_booklet(8, [8], bad)


def test_page_count_limits():
    with pytest.raises(BookletError, match="正整数"):
        plan_booklet(0, [], 1)
    with pytest.raises(BookletError, match="超过上限"):
        plan_booklet(MAX_PAGES + 1, [MAX_PAGES + 1], 1)
    # 上限边界可用
    result = plan_booklet(MAX_PAGES, [MAX_PAGES], 125)
    assert result.signature_count == 1


# ---------------------------------------------------------------- 随机对照


def test_random_against_brute_force():
    rng = random.Random(20260925)
    for _ in range(400):
        page_count = rng.randint(1, 12)
        cuts = sorted(
            rng.sample(range(1, page_count), rng.randint(0, page_count - 1))
        )
        ends = cuts + [page_count]
        max_sheets = rng.randint(1, 3)
        if _brute_force(ends, max_sheets) is None:
            with pytest.raises(BookletError):
                plan_booklet(page_count, ends, max_sheets)
        else:
            _check_against_brute_force(page_count, ends, max_sheets)


def test_random_single_section_cases():
    rng = random.Random(7)
    for _ in range(50):
        page_count = rng.randint(1, 16)
        max_sheets = rng.randint(1, 4)
        need = (page_count + 3) // 4
        if need > max_sheets:
            with pytest.raises(BookletError):
                plan_booklet(page_count, [page_count], max_sheets)
        else:
            _check_against_brute_force(page_count, [page_count], max_sheets)


def test_all_partitions_small_book_exhaustive():
    # 8 页的全部 2^7 种曲段划分 × 印张上限 1..3，逐一与暴力对照
    for max_sheets in (1, 2, 3):
        for r in range(0, 8):
            for cuts in itertools.combinations(range(1, 8), r):
                ends = list(cuts) + [8]
                if _brute_force(ends, max_sheets) is None:
                    with pytest.raises(BookletError):
                        plan_booklet(8, ends, max_sheets)
                else:
                    _check_against_brute_force(8, ends, max_sheets)


def test_large_input_performance():
    # 500 段（每段 1 页）、每帖至多 125 张印张：O(m²) DP 应迅速完成
    ends = list(range(1, MAX_PAGES + 1))
    result = plan_booklet(MAX_PAGES, ends, 125)
    assert result.signature_count == 1
    assert result.total_blank_pages == (-MAX_PAGES) % 4
    _check_layout(result)
