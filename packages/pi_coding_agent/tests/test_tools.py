"""Tests for the built-in coding tools (read/write/edit/bash/grep/find/ls)."""

from __future__ import annotations

import asyncio

import pytest

from pi_coding_agent.tools import create_default_tools

pytestmark = pytest.mark.asyncio


def _tools(tmp_path):
    return {t.name: t for t in create_default_tools(str(tmp_path))}


async def test_read(tmp_path):
    (tmp_path / "a.txt").write_text("hello\nworld\n")
    res = await _tools(tmp_path)["read"].execute("id", {"file_path": "a.txt"}, None, None)
    assert "hello" in res.content[0].text


async def test_read_offset_limit(tmp_path):
    (tmp_path / "a.txt").write_text("l0\nl1\nl2\nl3\n")
    res = await _tools(tmp_path)["read"].execute("id", {"file_path": "a.txt", "offset": 1, "limit": 2}, None, None)
    assert res.content[0].text.splitlines() == ["l1", "l2"]


async def test_read_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        await _tools(tmp_path)["read"].execute("id", {"file_path": "nope.txt"}, None, None)


async def test_write_and_edit(tmp_path):
    tools = _tools(tmp_path)
    await tools["write"].execute("id", {"file_path": "b.txt", "content": "foo bar baz"}, None, None)
    assert (tmp_path / "b.txt").read_text() == "foo bar baz"
    await tools["edit"].execute("id", {"file_path": "b.txt", "old_string": "bar", "new_string": "QUX"}, None, None)
    assert (tmp_path / "b.txt").read_text() == "foo QUX baz"


async def test_edit_old_string_missing(tmp_path):
    (tmp_path / "c.txt").write_text("abc")
    with pytest.raises(ValueError):
        await _tools(tmp_path)["edit"].execute("id", {"file_path": "c.txt", "old_string": "zzz", "new_string": "x"}, None, None)


async def test_bash(tmp_path):
    res = await _tools(tmp_path)["bash"].execute("id", {"command": "echo hi"}, None, None)
    assert "hi" in res.content[0].text
    assert res.details["exit_code"] == 0


async def test_bash_nonzero_exit(tmp_path):
    res = await _tools(tmp_path)["bash"].execute("id", {"command": "exit 3"}, None, None)
    assert res.details["exit_code"] == 3


async def test_grep(tmp_path):
    (tmp_path / "x.py").write_text("import os\n# TODO fix this\n")
    res = await _tools(tmp_path)["grep"].execute("id", {"pattern": "TODO", "path": "."}, None, None)
    assert "TODO" in res.content[0].text
    res_none = await _tools(tmp_path)["grep"].execute("id", {"pattern": "NOPE_NOPE"}, None, None)
    assert "No matches" in res_none.content[0].text


async def test_find(tmp_path):
    (tmp_path / "y.py").write_text("")
    res = await _tools(tmp_path)["find"].execute("id", {"pattern": "*.py"}, None, None)
    assert "y.py" in res.content[0].text


async def test_ls(tmp_path):
    (tmp_path / "z.txt").write_text("")
    res = await _tools(tmp_path)["ls"].execute("id", {"path": "."}, None, None)
    assert "z.txt" in res.content[0].text
