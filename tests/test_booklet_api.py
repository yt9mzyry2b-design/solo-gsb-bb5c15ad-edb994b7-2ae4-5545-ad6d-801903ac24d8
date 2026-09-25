"""纸质乐谱小册子拼版接口的端到端测试（FastAPI TestClient + 临时 SQLite）。"""

from __future__ import annotations

import importlib
import sys

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """以临时 DATA_DIR 重新加载 app.main，得到隔离的服务实例。"""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    sys.modules.pop("app.main", None)
    main = importlib.import_module("app.main")
    with TestClient(main.app) as c:
        yield c


def _create_plan(client, measure_count=8) -> str:
    """创建并保存一个规划：每小节高度 1、容量 1，每小节一页。"""
    resp = client.post(
        "/api/v1/plans",
        json={
            "measure_heights": [1] * measure_count,
            "silence_seconds": [5] * measure_count,
            "page_height": 1,
            "min_turn_time": 1,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def _impose(client, plan_id, ends, max_sheets):
    return client.post(
        f"/api/v1/plans/{plan_id}/booklet-imposition",
        json={
            "section_end_pages": ends,
            "max_sheets_per_signature": max_sheets,
        },
    )


def test_impose_single_signature_exact_layout(client):
    plan_id = _create_plan(client, measure_count=8)
    resp = _impose(client, plan_id, [3, 8], 2)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["id"] == plan_id
    assert body["request"] == {
        "section_end_pages": [3, 8],
        "max_sheets_per_signature": 2,
    }
    assert body["page_count"] == 8
    # 两段合为一帖（8 页恰 2 张印张、0 空白），优于分两帖的 4 页空白
    assert body["signature_count"] == 1
    assert body["total_blank_pages"] == 0
    (sig,) = body["signatures"]
    assert sig["signature_index"] == 1
    assert sig["start_page"] == 1
    assert sig["end_page"] == 8
    assert sig["blank_pages"] == 0
    assert sig["sheet_count"] == 2
    assert sig["sheets"] == [
        {
            "sheet_index": 1,
            "front": {"left": 8, "right": 1},
            "back": {"left": 2, "right": 7},
        },
        {
            "sheet_index": 2,
            "front": {"left": 6, "right": 3},
            "back": {"left": 4, "right": 5},
        },
    ]


def test_impose_blank_pages_are_null(client):
    plan_id = _create_plan(client, measure_count=6)
    resp = _impose(client, plan_id, [6], 2)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total_blank_pages"] == 2
    (sig,) = body["signatures"]
    assert sig["blank_pages"] == 2
    assert sig["sheets"][0]["front"] == {"left": None, "right": 1}
    assert sig["sheets"][0]["back"] == {"left": 2, "right": None}
    assert sig["sheets"][1]["front"] == {"left": 6, "right": 3}
    assert sig["sheets"][1]["back"] == {"left": 4, "right": 5}


def test_impose_multiple_signatures_lexicographic(client):
    # 三段各 2 页、每帖至多 1 张印张：[1..2][3..6] 与 [1..4][5..6]
    # 总空白同为 2、帖数同为 2，取末页序列字典序最小的 [2, 6]
    plan_id = _create_plan(client, measure_count=6)
    resp = _impose(client, plan_id, [2, 4, 6], 1)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["signature_count"] == 2
    assert body["total_blank_pages"] == 2
    assert [s["end_page"] for s in body["signatures"]] == [2, 6]
    first, second = body["signatures"]
    assert (first["start_page"], first["end_page"], first["blank_pages"]) == (
        1,
        2,
        2,
    )
    assert (second["start_page"], second["end_page"], second["blank_pages"]) == (
        3,
        6,
        0,
    )
    assert first["sheets"] == [
        {
            "sheet_index": 1,
            "front": {"left": None, "right": 1},
            "back": {"left": 2, "right": None},
        }
    ]
    assert second["sheets"] == [
        {
            "sheet_index": 1,
            "front": {"left": 6, "right": 3},
            "back": {"left": 4, "right": 5},
        }
    ]


def test_impose_plan_not_found_returns_404(client):
    resp = _impose(client, "does-not-exist", [1], 1)
    assert resp.status_code == 404
    assert "规划结果不存在" in resp.json()["detail"]


def test_impose_single_section_over_capacity_returns_422(client):
    plan_id = _create_plan(client, measure_count=8)
    # 单段 8 页需 2 张印张，每帖至多 1 张：无解
    resp = _impose(client, plan_id, [8], 1)
    assert resp.status_code == 422, resp.text
    detail = resp.json()["detail"]
    assert "超过每帖最多" in detail
    assert "曲段不可跨帖" in detail


def test_impose_middle_section_over_capacity_returns_422(client):
    plan_id = _create_plan(client, measure_count=8)
    resp = _impose(client, plan_id, [1, 6, 8], 1)
    assert resp.status_code == 422, resp.text
    assert "第 2 段" in resp.json()["detail"]


@pytest.mark.parametrize(
    "ends,fragment",
    [
        ([], "不能为空"),
        ([4], "必须等于方案页数"),
        ([2, 4], "必须等于方案页数"),
        ([8, 4], "严格递增"),
        ([4, 4], "严格递增"),
        ([0, 8], "越界"),
        ([9], "越界"),
    ],
)
def test_impose_invalid_section_ends_return_422(client, ends, fragment):
    plan_id = _create_plan(client, measure_count=8)
    resp = _impose(client, plan_id, ends, 2)
    assert resp.status_code == 422, resp.text
    assert fragment in resp.text


@pytest.mark.parametrize("bad", [1.5, 2.0, True, False, "2", None])
def test_impose_non_integer_end_pages_return_422(client, bad):
    plan_id = _create_plan(client, measure_count=8)
    resp = _impose(client, plan_id, [2, bad, 8], 2)
    assert resp.status_code == 422, resp.text
    assert "必须为整数" in resp.text


@pytest.mark.parametrize("bad", [0, -1, 1.5, 2.0, True, "2", None])
def test_impose_invalid_max_sheets_return_422(client, bad):
    plan_id = _create_plan(client, measure_count=8)
    resp = _impose(client, plan_id, [8], bad)
    assert resp.status_code == 422, resp.text


def test_impose_missing_and_extra_fields_rejected(client):
    plan_id = _create_plan(client, measure_count=8)
    missing = client.post(
        f"/api/v1/plans/{plan_id}/booklet-imposition",
        json={"section_end_pages": [8]},
    )
    assert missing.status_code == 422
    extra = client.post(
        f"/api/v1/plans/{plan_id}/booklet-imposition",
        json={
            "section_end_pages": [8],
            "max_sheets_per_signature": 2,
            "notes": "x",
        },
    )
    assert extra.status_code == 422


def test_impose_plan_page_count_over_limit_returns_422(client):
    # 501 小节的规划产生 501 页，超过拼版上限 500
    plan_id = _create_plan(client, measure_count=501)
    resp = _impose(client, plan_id, [501], 126)
    assert resp.status_code == 422, resp.text
    assert "超过上限" in resp.json()["detail"]


def test_impose_plan_page_count_at_limit_accepted(client):
    plan_id = _create_plan(client, measure_count=500)
    resp = _impose(client, plan_id, [500], 125)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["signature_count"] == 1
    assert body["total_blank_pages"] == 0
    assert body["signatures"][0]["sheet_count"] == 125


def test_impose_does_not_modify_saved_plan(client):
    plan_id = _create_plan(client, measure_count=8)
    before = client.get(f"/api/v1/plans/{plan_id}").json()
    resp = _impose(client, plan_id, [4, 8], 2)
    assert resp.status_code == 200
    after = client.get(f"/api/v1/plans/{plan_id}").json()
    assert before == after
