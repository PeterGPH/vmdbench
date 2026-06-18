"""
retrieval_bridge.py — wrap a VMD tool bridge with error-triggered ("reactive") retrieval.

Our ablation showed the agent never calls a docs_search tool — not on its own, not even
when ordered to. So instead of relying on the model to retrieve, this wrapper does it in
the HARNESS: when run_vmd_command returns a Tcl error, it queries the local docs index for
the offending command and appends the correct syntax to the error the model sees on the
next turn. Retrieval becomes automatic and action-triggered (DocPrompting/FLARE-style),
with no docs_search tool exposed to the model.

Duck-typed stand-in for the bridge: execute_tool / reset / close delegate to the wrapped
bridge; only failing run_vmd_command results are augmented.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

# VMD commands worth keying a lookup on (the ones whose syntax the model gets wrong).
_CMD_RE = re.compile(r"\b(measure\s+\w+|atomselect|mol\s+new|mol\s+\w+|pbc\s+\w+)", re.I)


class RetrievalAugmentingBridge:
    """Wraps an inner bridge; on a Tcl error, appends retrieved reference syntax."""

    def __init__(self, inner: Any, docs_search: Any = None,
                 max_hint_chars: int = 700, k: int = 1):
        self._inner = inner
        self._docs = docs_search
        self._max = int(max_hint_chars)
        self._k = int(k)
        self._avail = bool(getattr(docs_search, "is_available", False))
        self.hint_count = 0   # how many times we injected (for debugging/metrics)

    # ----------------------------------------------------------- passthrough
    def reset(self, *a, **k):
        return self._inner.reset(*a, **k)

    def close(self, *a, **k):
        if hasattr(self._inner, "close"):
            return self._inner.close(*a, **k)
        return None

    def __getattr__(self, name):  # forward anything else to the inner bridge
        return getattr(self._inner, name)

    # ------------------------------------------------------------ tool entry
    def execute_tool(self, *, tool_name: str = "",
                     tool_input: Optional[Dict[str, Any]] = None, **kw) -> Dict[str, Any]:
        res = self._inner.execute_tool(tool_name=tool_name, tool_input=tool_input, **kw)
        try:
            if (tool_name == "run_vmd_command" and isinstance(res, dict)
                    and not res.get("ok") and self._avail):
                hint = self._lookup(str((tool_input or {}).get("command") or ""),
                                    str(res.get("error") or ""))
                if hint:
                    self.hint_count += 1
                    res["error"] = (str(res.get("error") or "").rstrip()
                                    + "\n\n[auto-retrieved VMD reference — use this exact form]\n"
                                    + hint)
        except Exception:
            pass  # augmentation must never break a run
        return res

    # -------------------------------------------------------------- internals
    def _lookup(self, command: str, error: str) -> str:
        # The error usually names the failing command ("measure dihed: ...").
        q = ""
        head = error.split(":", 1)[0].strip()
        if head and len(head) <= 40:
            q = head
        m = _CMD_RE.search(command)
        if m:
            q = (q + " " + m.group(0)).strip()
        if not q:
            q = (error or command)[:80]
        try:
            r = self._docs.search(q, k=self._k, scope="all")
        except Exception:
            return ""
        if not r.get("ok") or not r.get("results"):
            return ""
        text = (r["results"][0].get("text") or "").strip()
        return (text[:self._max] + " …") if len(text) > self._max else text
