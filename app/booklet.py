"""纸质乐谱小册子拼版核心。

背景
====
一份已保存的翻页方案把乐谱分成 page_count 页（页码 1..page_count）。把纸质谱
装订成骑马订小册子时，整本谱按**曲段**（例如乐章）切分：请求给出严格递增的
各曲段**末页页码**，末项必须等于方案页数——曲段因此是 1..page_count 的有序
完整划分。若干相邻曲段合成一**帖**（signature）印刷装订，约束：

- 曲段不可跨帖，每帖至少含一段；
- 一张印张正反面共 4 页，故每帖页数（含空白）必须是 4 的倍数，不足时在
  **帖尾补空白页**；
- 每帖最多 max_sheets 张印张，即至多容纳 4·max_sheets 页；单个曲段自身
  超过该容量时不存在任何合法分帖（各段独立成帖即合法，故这是唯一无解
  情形），直接报错。

在所有合法分帖中依次最小化：(1) **总空白页数**；(2) **帖数**；仍并列时取
**各帖末页（原谱页码）序列字典序最小**的方案，使结果确定、可复现。

动态规划
========
曲段数 m（≤ 页数 ≤ MAX_PAGES）。一帖容纳第 a..b 段（0 基闭区间）时，页数
为区间末页差 P = E[b] − E[a−1]，需 ceil(P/4) 张印张，合法当且仅当不超过
max_sheets，空白为 4·ceil(P/4) − P。

- 前向 DP：``fwd[j]`` = 前 j 段的最优 (总空白, 帖数)，按字典序取 min；
- 后缀 DP：``suf[a]`` = 第 a..m−1 段的最优 (总空白, 帖数)；
- 字典序最小重构：``fwd[pos] + suf[pos] == 全局最优`` 在重构过程中始终
  成立（归纳易证），故"当前前缀能扩展成某个全局最优方案"可由
  ``fwd[pos] + 本帖代价 + suf[b+1] == 全局最优`` 精确刻画。自左向右贪心，
  每次取满足该式的**最小**帖末下标 b——末页随 b 严格递增，逐位取最小即得
  末页序列字典序最小的方案。

每次转移枚举 O(m) 个候选，整体 O(m²)。

骑马订拼版
==========
一帖 S 张印张共 4S 个页位：原谱页按顺序占据前 P 个页位，尾部 4S−P 个为空白
（响应中以 null 表示）。从外到内第 k 张印张（1 基）：正面左页为倒数第
2k−1 个页位、右页为正数第 2k−1 个页位；背面左页为正数第 2k 个、右页为
倒数第 2k 个。即最外层印张正面为 末页/首页、背面为 次页/倒数第二页，向内
逐层交替；同一张印张上每两页页位下标之和恒为 4S−1。

本模块仅依赖标准库，便于脱离 Web 层直接单测。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

#: 规划页数上限（与纸质备用谱页序核对的页数上限一致）
MAX_PAGES = 500


class BookletError(ValueError):
    """小册子拼版输入不合法：曲段末页序列非法、印张上限非法或单段超容量等。"""


@dataclass(frozen=True)
class SheetSide:
    """印张一面的左右页码；空白页为 None。"""

    left: Optional[int]
    right: Optional[int]


@dataclass(frozen=True)
class Sheet:
    """一张印张的正反两面。"""

    sheet_index: int  # 印张序号（1 基，从外到内）
    front: SheetSide
    back: SheetSide


@dataclass(frozen=True)
class Signature:
    """一帖：连续若干曲段的原谱页范围、帖尾空白与骑马订拼版结果。"""

    signature_index: int  # 帖序号（1 基）
    start_page: int  # 本帖原谱首页（1 基）
    end_page: int  # 本帖原谱末页（1 基）
    blank_pages: int  # 帖尾空白页数（补到 4 的倍数）
    sheet_count: int  # 本帖印张数
    sheets: List[Sheet]  # 从外到内各印张的正反面左右页码


@dataclass(frozen=True)
class BookletPlan:
    """小册子拼版结果。"""

    page_count: int  # 方案页数
    signature_count: int  # 帖数
    total_blank_pages: int  # 总空白页数（各帖帖尾空白之和）
    signatures: List[Signature]  # 按谱面顺序的各帖


def _validate(
    page_count: int,
    section_end_pages: Sequence[int],
    max_sheets_per_signature: int,
) -> List[int]:
    """校验输入，返回曲段末页页码列表。

    :raises BookletError: 任一约束不满足
    """
    if not isinstance(page_count, int) or isinstance(page_count, bool):
        raise BookletError("规划页数必须为整数")
    if page_count < 1:
        raise BookletError(f"规划页数必须为正整数（实际 {page_count}）")
    if page_count > MAX_PAGES:
        raise BookletError(
            f"规划页数（{page_count}）超过上限 {MAX_PAGES}，无法拼版"
        )

    if not isinstance(max_sheets_per_signature, int) or isinstance(
        max_sheets_per_signature, bool
    ):
        raise BookletError(
            f"每帖最多印张数必须为整数（收到 {max_sheets_per_signature!r}）"
        )
    if max_sheets_per_signature < 1:
        raise BookletError(
            f"每帖最多印张数必须为正整数（实际 {max_sheets_per_signature}）"
        )

    if not isinstance(section_end_pages, Sequence) or isinstance(
        section_end_pages, (str, bytes)
    ):
        raise BookletError("section_end_pages 必须是曲段末页页码数组")
    if not section_end_pages:
        raise BookletError(
            "section_end_pages 不能为空：至少需要一个曲段"
            "（末项须等于方案页数）"
        )

    ends: List[int] = []
    prev = 0
    for idx, end in enumerate(section_end_pages):
        if not isinstance(end, int) or isinstance(end, bool):
            raise BookletError(
                f"第 {idx + 1} 个曲段末页页码必须为整数（收到 {end!r}）"
            )
        if end < 1 or end > page_count:
            raise BookletError(
                f"第 {idx + 1} 个曲段末页页码 {end} 越界："
                f"合法范围为 1..{page_count}"
            )
        if end <= prev:
            raise BookletError(
                f"曲段末页页码必须严格递增：第 {idx + 1} 项（{end}）"
                f"不大于前一项（{prev}）"
            )
        ends.append(end)
        prev = end
    if ends[-1] != page_count:
        raise BookletError(
            f"曲段末页末项（{ends[-1]}）必须等于方案页数（{page_count}）："
            "曲段是整本谱的完整划分"
        )

    # 单段容量：曲段不可跨帖，任何一段自身所需印张数超过每帖上限即无解
    # （反之各段独立成帖必合法，故这是唯一的无解情形）。
    prev = 0
    for idx, end in enumerate(ends):
        length = end - prev
        need = (length + 3) // 4
        if need > max_sheets_per_signature:
            raise BookletError(
                f"第 {idx + 1} 段（第 {prev + 1}..{end} 页，共 {length} 页）"
                f"至少需要 {need} 张印张，超过每帖最多 "
                f"{max_sheets_per_signature} 张，曲段不可跨帖，无法拼版"
            )
        prev = end
    return ends


def plan_booklet(
    page_count: int,
    section_end_pages: Sequence[int],
    max_sheets_per_signature: int,
) -> BookletPlan:
    """为已保存翻页方案做纸质乐谱小册子拼版。

    :param page_count: 方案页数（页码范围 1..page_count），上限 MAX_PAGES
    :param section_end_pages: 严格递增的各曲段末页页码（1 基），非空、
        末项等于 page_count
    :param max_sheets_per_signature: 每帖最多印张数（正整数，每张 4 页）
    :return: 各帖的原谱页范围、帖尾空白页数与骑马订拼版页序
    :raises BookletError: 末页序列非法、印张上限非法或单段超容量等
    """
    ends = _validate(page_count, section_end_pages, max_sheets_per_signature)
    max_sheets = max_sheets_per_signature
    m = len(ends)

    def signature_blanks(a: int, b: int) -> Optional[int]:
        """一帖容纳第 a..b 段（0 基闭区间）时的帖尾空白页数；超容量为 None。"""
        length = ends[b] - (ends[a - 1] if a > 0 else 0)
        if (length + 3) // 4 > max_sheets:
            return None
        return (-length) % 4

    # 前向 DP：fwd[j] = 前 j 段的最优 (总空白, 帖数)；None 表示不可达。
    fwd: List[Optional[Tuple[int, int]]] = [None] * (m + 1)
    fwd[0] = (0, 0)
    for j in range(1, m + 1):
        best: Optional[Tuple[int, int]] = None
        for a in range(j):
            fa = fwd[a]
            if fa is None:
                continue
            blanks = signature_blanks(a, j - 1)
            if blanks is None:
                continue
            cand = (fa[0] + blanks, fa[1] + 1)
            if best is None or cand < best:
                best = cand
        fwd[j] = best

    # 后缀 DP：suf[a] = 第 a..m-1 段的最优 (总空白, 帖数)。
    suf: List[Optional[Tuple[int, int]]] = [None] * (m + 1)
    suf[m] = (0, 0)
    for a in range(m - 1, -1, -1):
        best = None
        for b in range(a, m):
            blanks = signature_blanks(a, b)
            if blanks is None:
                continue
            sb = suf[b + 1]
            if sb is None:
                continue
            cand = (blanks + sb[0], 1 + sb[1])
            if best is None or cand < best:
                best = cand
        suf[a] = best

    total = fwd[m]
    assert total is not None, "内部错误：单段容量已校验，必存在合法分帖"

    # 字典序最小末页序列重构：不变式 fwd[pos] + suf[pos] == total 保证当前
    # 前缀可扩展成全局最优；每次取仍保持全局最优的最小帖末下标 b（末页随
    # b 严格递增，即末页最小），逐位最小即末页序列字典序最小。
    groups: List[Tuple[int, int]] = []
    pos = 0
    while pos < m:
        fp = fwd[pos]
        assert fp is not None, "内部错误：拼版方案重构失败"
        for b in range(pos, m):
            blanks = signature_blanks(pos, b)
            if blanks is None:
                continue
            sb = suf[b + 1]
            if sb is None:
                continue
            if (fp[0] + blanks + sb[0], fp[1] + 1 + sb[1]) == total:
                groups.append((pos, b))
                pos = b + 1
                break
        else:
            raise AssertionError("内部错误：拼版方案重构失败")

    # 拼装每帖的骑马订页序：页位 0..4S-1 中前 length 个为原谱页，其余空白。
    signatures: List[Signature] = []
    for idx, (a, b) in enumerate(groups, start=1):
        start_page = (ends[a - 1] if a > 0 else 0) + 1
        end_page = ends[b]
        length = end_page - start_page + 1
        sheet_count = (length + 3) // 4
        slot_count = sheet_count * 4
        blanks = slot_count - length
        slots: List[Optional[int]] = [
            start_page + i if i < length else None for i in range(slot_count)
        ]
        sheets: List[Sheet] = []
        for k in range(1, sheet_count + 1):
            sheets.append(
                Sheet(
                    sheet_index=k,
                    front=SheetSide(
                        left=slots[slot_count - 2 * k + 1],
                        right=slots[2 * k - 2],
                    ),
                    back=SheetSide(
                        left=slots[2 * k - 1],
                        right=slots[slot_count - 2 * k],
                    ),
                )
            )
        signatures.append(
            Signature(
                signature_index=idx,
                start_page=start_page,
                end_page=end_page,
                blank_pages=blanks,
                sheet_count=sheet_count,
                sheets=sheets,
            )
        )

    # 防御：逐帖重算的空白数与帖数必须等于 DP 最优值。
    assert sum(s.blank_pages for s in signatures) == total[0], (
        "内部错误：重构空白数与最优空白数不一致"
    )
    assert len(signatures) == total[1], "内部错误：重构帖数与最优帖数不一致"

    return BookletPlan(
        page_count=page_count,
        signature_count=len(signatures),
        total_blank_pages=total[0],
        signatures=signatures,
    )
