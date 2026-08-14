"""Built-in coding tools (Python port of packages/coding-agent/src/core/tools/*).

Each tool is an :class:`~pi_agent_core.types.AgentTool` whose ``execute``
coroutine has the signature the agent loop expects::

    async def execute(tool_call_id, args, signal, on_update) -> AgentToolResult

Errors are signalled by **raising** (the core loop catches the exception and
marks the tool result as an error). Successful output is returned as an
``AgentToolResult`` carrying ``TextContent`` blocks.

Paths are resolved relative to a workspace ``cwd`` and a best-effort guard
refuses to escape it. **Production use must run these inside the external
Docker sandbox** (per the migration decision) — this module only shells out to
the local process; it does not itself provide isolation.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any, Callable, List, Optional

from pi_agent_core.types import AgentTool, AgentToolResult
from pi_ai import TextContent

__all__ = ["create_default_tools", "AgentTool", "AgentToolResult"]


def _result(text: str) -> AgentToolResult:
    return AgentToolResult(content=[TextContent(text=text)])


def _safe_path(cwd: str, p: str) -> Path:
    """Resolve ``p`` against ``cwd`` and refuse to escape the workspace.

    Production isolation is delegated to the external Docker sandbox; this is a
    defensive belt-and-suspenders check, not a security boundary.
    """
    base = Path(cwd).resolve()
    target = (base / p).resolve()
    if target != base and base not in target.parents:
        raise ValueError(f"Path escapes workspace root: {p!r}")
    return target


# ---------------------------------------------------------------------------
# read / write / edit
# ---------------------------------------------------------------------------
async def _read_execute(cwd: str, tool_call_id, args, signal, on_update):
    path = _safe_path(cwd, args["file_path"])
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {args['file_path']}")
    text = path.read_text(encoding="utf-8", errors="replace")
    offset = args.get("offset")
    limit = args.get("limit")
    if offset is not None or limit is not None:
        lines = text.splitlines(keepends=True)
        o = offset or 0
        lines = lines[o : o + limit] if limit else lines[o:]
        text = "".join(lines)
    if on_update:
        on_update(text)
    return _result(text)


async def _write_execute(cwd: str, tool_call_id, args, signal, on_update):
    path = _safe_path(cwd, args["file_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    content = args["content"]
    path.write_text(content, encoding="utf-8")
    msg = f"Wrote {len(content)} characters to {args['file_path']}"
    if on_update:
        on_update(msg)
    return _result(msg)


async def _edit_execute(cwd: str, tool_call_id, args, signal, on_update):
    path = _safe_path(cwd, args["file_path"])
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {args['file_path']}")
    old = args["old_string"]
    new = args["new_string"]
    if not old:
        raise ValueError("old_string must be non-empty")
    content = path.read_text(encoding="utf-8", errors="replace")
    if old not in content:
        raise ValueError(f"old_string not found in {args['file_path']}")
    content = content.replace(old, new) if args.get("replace_all") else content.replace(old, new, 1)
    path.write_text(content, encoding="utf-8")
    msg = f"Edited {args['file_path']}"
    if on_update:
        on_update(msg)
    return _result(msg)


# ---------------------------------------------------------------------------
# bash
# ---------------------------------------------------------------------------
async def _bash_execute(cwd: str, tool_call_id, args, signal, on_update):
    command = args["command"]
    timeout = args.get("timeout", 120)
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise TimeoutError(f"Command timed out after {timeout}s")
    text = (out or b"").decode(errors="replace")
    if err:
        text += "\n[stderr]\n" + (err or b"").decode(errors="replace")
    if on_update:
        on_update(text)
    return AgentToolResult(content=[TextContent(text=text)], details={"exit_code": proc.returncode})


# ---------------------------------------------------------------------------
# grep (Python regex walk — avoids a hard dependency on ripgrep)
# ---------------------------------------------------------------------------
async def _grep_execute(cwd: str, tool_call_id, args, signal, on_update):
    pattern = args["pattern"]
    root = _safe_path(cwd, args.get("path", "."))
    flags = re.IGNORECASE if args.get("ignore_case") else 0
    rx = re.compile(pattern, flags)
    glob = args.get("glob", "**/*")
    base = Path(cwd).resolve()
    matches: List[str] = []
    for f in root.glob(glob):
        if not f.is_file():
            continue
        try:
            lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        for i, line in enumerate(lines, 1):
            if rx.search(line):
                matches.append(f"{f.relative_to(base)}:{i}:{line}")
    text = "\n".join(matches) if matches else "No matches found."
    if on_update:
        on_update(text)
    return _result(text)


# ---------------------------------------------------------------------------
# find / ls
# ---------------------------------------------------------------------------
async def _find_execute(cwd: str, tool_call_id, args, signal, on_update):
    root = _safe_path(cwd, args.get("path", "."))
    base = Path(cwd).resolve()
    results = sorted(str(p.relative_to(base)) for p in root.glob(args["pattern"]))
    text = "\n".join(results) if results else "No files found."
    if on_update:
        on_update(text)
    return _result(text)


async def _ls_execute(cwd: str, tool_call_id, args, signal, on_update):
    root = _safe_path(cwd, args.get("path", "."))
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {args.get('path', '.')}")
    entries = sorted(
        (p.name + "/" if p.is_dir() else p.name) for p in root.iterdir()
    )
    text = "\n".join(entries) if entries else "(empty)"
    if on_update:
        on_update(text)
    return _result(text)


# ---------------------------------------------------------------------------
# Tool definitions (JSON schema + execute closure bound to cwd)
# ---------------------------------------------------------------------------
def _make(
    name: str,
    description: str,
    parameters: dict,
    execute: Callable[..., Any],
    cwd: str,
) -> AgentTool:
    async def _bound(tool_call_id, args, signal, on_update):
        return await execute(cwd, tool_call_id, args, signal, on_update)

    return AgentTool(
        name=name,
        description=description,
        parameters=parameters,
        label=name,
        execute=_bound,
    )


_READ_PARAMS = {
    "type": "object",
    "properties": {
        "file_path": {"type": "string", "description": "Path to the file to read, relative to the workspace."},
        "offset": {"type": "integer", "description": "Line offset to start reading from (0-based)."},
        "limit": {"type": "integer", "description": "Maximum number of lines to read."},
    },
    "required": ["file_path"],
}

_WRITE_PARAMS = {
    "type": "object",
    "properties": {
        "file_path": {"type": "string", "description": "Path to the file to write, relative to the workspace."},
        "content": {"type": "string", "description": "Full file content to write."},
    },
    "required": ["file_path", "content"],
}

_EDIT_PARAMS = {
    "type": "object",
    "properties": {
        "file_path": {"type": "string", "description": "Path to the file to edit, relative to the workspace."},
        "old_string": {"type": "string", "description": "Exact text to replace."},
        "new_string": {"type": "string", "description": "Replacement text."},
        "replace_all": {"type": "boolean", "description": "Replace all occurrences instead of just the first."},
    },
    "required": ["file_path", "old_string", "new_string"],
}

_BASH_PARAMS = {
    "type": "object",
    "properties": {
        "command": {"type": "string", "description": "Shell command to execute."},
        "timeout": {"type": "integer", "description": "Timeout in seconds (default 120)."},
    },
    "required": ["command"],
}

_GREP_PARAMS = {
    "type": "object",
    "properties": {
        "pattern": {"type": "string", "description": "Regular expression to search for."},
        "path": {"type": "string", "description": "Directory to search in (default '.')."},
        "glob": {"type": "string", "description": "Glob filter for files (default '**/*')."},
        "ignore_case": {"type": "boolean", "description": "Case-insensitive match."},
    },
    "required": ["pattern"],
}

_FIND_PARAMS = {
    "type": "object",
    "properties": {
        "pattern": {"type": "string", "description": "Glob pattern, e.g. '*.py'."},
        "path": {"type": "string", "description": "Directory to search in (default '.')."},
    },
    "required": ["pattern"],
}

_LS_PARAMS = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Directory to list (default '.')."},
    },
    "required": [],
}


def create_default_tools(cwd: str) -> List[AgentTool]:
    """Return the default built-in tool set bound to ``cwd``.

    Mirrors the upstream coding-agent kit: ``read`` / ``write`` / ``edit`` /
    ``bash`` / ``grep`` / ``find`` / ``ls``.
    """
    return [
        _make("read", "Read a file from the workspace.", _READ_PARAMS, _read_execute, cwd),
        _make("write", "Write (create or overwrite) a file in the workspace.", _WRITE_PARAMS, _write_execute, cwd),
        _make("edit", "Replace text in a file (str-replace).", _EDIT_PARAMS, _edit_execute, cwd),
        _make("bash", "Run a shell command in the workspace (sandbox via Docker in production).", _BASH_PARAMS, _bash_execute, cwd),
        _make("grep", "Search file contents with a regex.", _GREP_PARAMS, _grep_execute, cwd),
        _make("find", "Find files matching a glob.", _FIND_PARAMS, _find_execute, cwd),
        _make("ls", "List a directory's entries.", _LS_PARAMS, _ls_execute, cwd),
    ]
