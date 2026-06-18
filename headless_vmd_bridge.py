"""
headless_vmd_bridge.py — a drop-in replacement for VmdToolBridge that executes
VMD tool calls in an *in-process* headless VMD (vmd-python) instead of
round-tripping to a live Tcl/Tk panel.

Why this exists
---------------
``ClaudeToolLoop.run()`` dispatches every VMD tool call through:

    tool_bridge.execute_tool(session_id=..., tool_call_id=..., tool_name=...,
                             tool_input=..., session_queue=..., cancel_event=...)
        -> {"ok": bool, "output": str, "error": str}        # run_vmd_command
        -> {... , "image_b64": str, "image_mime": str}       # capture_vmd_snapshot

That is the *only* coupling between the loop and VMD. So to run the agent
head­lessly inside SciVisAgentBench we just supply an object with the same
``execute_tool`` signature. This one uses ``vmd-python`` (``import vmd``):

  * ``run_vmd_command``     -> vmd.evaltcl(command), with C-level stdout captured
                              so the model still sees ``puts`` output.
  * ``capture_vmd_snapshot``-> best-effort render (vmd-python has NO
                              TachyonInternal; molecular_vis is text-graded so
                              the image is for the agent's own verification).

Reset between tasks: vmd-python is a process-global interpreter, so call
``reset()`` at the start of each task to clear molecules/state.
"""
from __future__ import annotations

import base64
import contextlib
import os
import tempfile
import threading
from typing import Any, Dict, Optional


class HeadlessVmdBridge:
    """Duck-typed stand-in for runtime VmdToolBridge, backed by vmd-python."""

    def __init__(self) -> None:
        self._evaltcl = None  # lazily bound vmd.evaltcl
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ setup
    def _ensure_vmd(self):
        if self._evaltcl is None:
            try:
                from vmd import evaltcl  # type: ignore
            except Exception as exc:  # pragma: no cover - import guard
                raise RuntimeError(
                    "vmd-python is not importable. Install it into the same "
                    "environment, e.g. `pip install vmd-python` (conda-forge "
                    "also ships it). Original error: %r" % (exc,)
                )
            self._evaltcl = evaltcl
        return self._evaltcl

    def reset(self) -> None:
        """Clear all molecules + view so each task starts from a clean scene."""
        evaltcl = self._ensure_vmd()
        with contextlib.suppress(Exception):
            evaltcl("foreach __m [molinfo list] { mol delete $__m }")
        with contextlib.suppress(Exception):
            evaltcl("display resetview")

    # -------------------------------------------------------------- tool entry
    def execute_tool(
        self,
        *,
        session_id: str = "",
        tool_call_id: str = "",
        tool_name: str = "",
        tool_input: Optional[Dict[str, Any]] = None,
        session_queue: Any = None,      # ignored — no Tcl round-trip
        cancel_event: Optional[threading.Event] = None,
        timeout: float = 45.0,
    ) -> Dict[str, Any]:
        tool_input = tool_input or {}
        if cancel_event is not None and cancel_event.is_set():
            return {"ok": False, "output": "", "error": "cancelled"}

        # vmd-python is not thread-safe; serialize tool calls.
        with self._lock:
            if tool_name == "run_vmd_command":
                return self._run_tcl(str(tool_input.get("command") or ""))
            if tool_name == "capture_vmd_snapshot":
                return self._snapshot(str(tool_input.get("purpose") or ""))
            return {
                "ok": False,
                "output": "",
                "error": f"HeadlessVmdBridge: unsupported tool {tool_name!r}",
            }

    # --------------------------------------------------------------- internals
    def _run_tcl(self, command: str) -> Dict[str, Any]:
        evaltcl = self._ensure_vmd()
        command = command.strip()
        if not command:
            return {"ok": False, "output": "", "error": "empty command"}
        try:
            with _capture_c_stdout() as cap:
                ret = evaltcl(command)
            printed = cap.read_text()
        except Exception as exc:  # Tcl error surfaces as a Python exception
            return {"ok": False, "output": "", "error": str(exc)}

        # Combine what `puts` wrote (printed) with the Tcl return value (ret),
        # mirroring what the live Tcl bridge would have sent back.
        parts = []
        if printed.strip():
            parts.append(printed.rstrip("\n"))
        if ret is not None and str(ret).strip():
            parts.append(str(ret).strip())
        output = "\n".join(parts) if parts else "(command ok, no output)"
        return {"ok": True, "output": output, "error": ""}

    def _snapshot(self, purpose: str) -> Dict[str, Any]:
        """Best-effort headless render. vmd-python lacks TachyonInternal, so
        we try the available renderers and fall back to a textual scene
        summary. Always ok=True so the agent does not burn turns retrying a
        capability the headless build does not have."""
        evaltcl = self._ensure_vmd()
        out_path = os.path.join(tempfile.gettempdir(), f"vmdai_snap_{os.getpid()}.tga")
        with contextlib.suppress(Exception):
            if os.path.exists(out_path):
                os.unlink(out_path)
        rendered = False
        for renderer in ("snapshot", "TachyonInternal"):
            try:
                evaltcl("display update")
                evaltcl(f'render {renderer} "{out_path}"')
            except Exception:
                continue
            if os.path.isfile(out_path) and os.path.getsize(out_path) > 1000:
                rendered = True
                break

        result: Dict[str, Any] = {"ok": True, "error": ""}
        if rendered:
            try:
                with open(out_path, "rb") as fh:
                    result["image_b64"] = base64.b64encode(fh.read()).decode()
                result["image_mime"] = "image/x-tga"
                result["output"] = f"Snapshot rendered ({purpose or 'viewport'})."
            except Exception:
                rendered = False
            finally:
                with contextlib.suppress(Exception):
                    os.unlink(out_path)
        if not rendered:
            # Textual fallback so the model still gets scene feedback.
            summary = ""
            with contextlib.suppress(Exception):
                summary = str(
                    evaltcl(
                        'set __s ""; foreach __m [molinfo list] { append __s '
                        '"mol $__m: [molinfo $__m get name], [molinfo $__m get numatoms] atoms, '
                        '[molinfo $__m get numreps] reps\\n" }; set __s'
                    )
                )
            result["output"] = (
                "Headless build: no raster snapshot available. "
                "Current scene:\n" + (summary or "(no molecules loaded)")
            )
        return result


@contextlib.contextmanager
def _capture_c_stdout():
    """Redirect C-level stdout (fd 1) to a temp file around a block, so that
    VMD's ``puts`` output (which bypasses Python's sys.stdout) is captured."""
    target_fd = 1
    saved_fd = os.dup(target_fd)
    tmp = tempfile.TemporaryFile(mode="w+b")
    os.dup2(tmp.fileno(), target_fd)

    class _Reader:
        def read_text(self) -> str:
            tmp.flush()
            tmp.seek(0)
            return tmp.read().decode("utf-8", "replace")

    try:
        yield _Reader()
    finally:
        os.dup2(saved_fd, target_fd)
        os.close(saved_fd)
        with contextlib.suppress(Exception):
            tmp.close()
