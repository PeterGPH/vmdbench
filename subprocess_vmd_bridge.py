"""
subprocess_vmd_bridge.py — headless VMD tool bridge backed by your *native*
VMD.app, driven as a persistent ``vmd -dispdev text`` process **through a
pseudo-terminal (pty)**.

Why a pty: VMD is an interactive Tcl/Tk console app. When its stdin/stdout are
plain pipes (not a terminal) it block-buffers stdout and may not read stdin as a
console — so a naive pipe driver hangs. A pty makes VMD behave exactly as it
does in a real terminal: it reads "typed" commands and line-buffers output.
(Unix/macOS only — which is your platform.)

No new install: uses the VMD binary you already have. Because it is full VMD,
``render TachyonInternal`` works, so snapshots are real images.

Interface (duck-typed stand-in for the runtime's VmdToolBridge):

    execute_tool(tool_name=..., tool_input=..., cancel_event=..., **_) -> {ok, output, error}
    reset()    # clear molecules between tasks
    close()    # terminate the VMD process

Per tool call we wrap the model's Tcl in ``catch`` (so a Tcl error never breaks
the stream), print sentinel markers, and read the pty until the DONE marker.
``puts`` output lands before the markers, so the model still sees what it printed.
"""
from __future__ import annotations

import base64
import contextlib
import os
import pty
import select
import shutil
import subprocess
import tempfile
import termios
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_RC = "@@VMDAI_RC@@"
_MB = "@@VMDAI_MSG_BEGIN@@"
_ME = "@@VMDAI_MSG_END@@"
_DONE = "@@VMDAI_DONE@@"
_PROMPT = "vmd >"
_MAX_TOOL_OUTPUT_CHARS = 6000   # cap tool_result size so verbose VMD output can't flood context


class _VmdTimeout(RuntimeError):
    pass


def _has_scripts(d: str) -> bool:
    return (Path(d) / "scripts" / "vmd" / "atomselmacros.dat").exists()


def _resolve_vmd(vmd_bin: Optional[str]) -> Tuple[str, Optional[str]]:
    """Return (binary, VMDDIR-or-None). Explicit override -> PATH binary with
    scripts alongside -> macOS app bundle (with injected VMDDIR)."""
    cand = vmd_bin or os.environ.get("VMD_AI_VMD_BIN") or os.environ.get("VMDBENCH_VMD_BIN")
    if cand:
        if not Path(cand).exists():
            raise RuntimeError(f"VMD binary {cand!r} does not exist")
        parent = str(Path(cand).parent)
        return cand, (parent if _has_scripts(parent) else None)

    onpath = shutil.which(vmd_bin or "vmd")
    if onpath and _has_scripts(str(Path(onpath).parent)):
        return onpath, None

    app = Path("/Applications/VMD.app/Contents/vmd")
    if app.exists():
        for b in sorted(app.glob("vmd_MACOSX*")):
            if os.access(b, os.X_OK):
                return str(b), str(app)
    if onpath:
        return onpath, None
    raise RuntimeError(
        "No VMD binary found. Set VMD_AI_VMD_BIN to your vmd_MACOSXARM64 "
        "(e.g. /Applications/VMD.app/Contents/vmd/vmd_MACOSXARM64)."
    )


class SubprocessVmdBridge:
    def __init__(self, vmd_bin: Optional[str] = None, timeout: int = 120):
        self._bin, self._vmddir = _resolve_vmd(vmd_bin)
        self._timeout = int(timeout)
        self._proc: Optional[subprocess.Popen] = None
        self._master_fd: Optional[int] = None
        self._buf = b""

    # ------------------------------------------------------------- lifecycle
    def _start(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        master_fd, slave_fd = pty.openpty()
        # Disable echo on the slave so the commands we write aren't echoed back
        # into the output stream we read.
        with contextlib.suppress(Exception):
            attrs = termios.tcgetattr(slave_fd)
            attrs[3] = attrs[3] & ~termios.ECHO          # lflags &= ~ECHO
            termios.tcsetattr(slave_fd, termios.TCSANOW, attrs)

        env = dict(os.environ)
        if self._vmddir:
            env["VMDDIR"] = self._vmddir

        self._proc = subprocess.Popen(
            [self._bin, "-dispdev", "text"],
            stdin=slave_fd, stdout=slave_fd, stderr=slave_fd,
            env=env, close_fds=True,
        )
        os.close(slave_fd)
        self._master_fd = master_fd
        self._buf = b""
        # Consume the startup banner via the protocol; fail fast if mis-set.
        self._exec_wrapped("set __vbai_ready 1", timeout=min(self._timeout, 45))

    def reset(self) -> None:
        self._start()
        self._exec_wrapped("foreach __m [molinfo list] { mol delete $__m }",
                           timeout=self._timeout)
        with contextlib.suppress(Exception):
            self._exec_wrapped("display resetview", timeout=30)

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            with contextlib.suppress(Exception):
                os.write(self._master_fd, b"quit\n")
            with contextlib.suppress(Exception):
                self._proc.wait(timeout=5)
            with contextlib.suppress(Exception):
                self._proc.kill()
        if self._master_fd is not None:
            with contextlib.suppress(Exception):
                os.close(self._master_fd)
        self._master_fd = None
        self._proc = None

    # ------------------------------------------------------------ tool entry
    def execute_tool(self, *, tool_name: str = "", tool_input: Optional[Dict[str, Any]] = None,
                     cancel_event: Optional[Any] = None, **_: Any) -> Dict[str, Any]:
        tool_input = tool_input or {}
        if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
            return {"ok": False, "output": "", "error": "cancelled"}
        try:
            self._start()
            if tool_name == "run_vmd_command":
                return self._run_tcl(str(tool_input.get("command") or ""))
            if tool_name == "capture_vmd_snapshot":
                return self._snapshot(str(tool_input.get("purpose") or ""),
                                      tool_input.get("save_path"))
            if tool_name == "vmd_measure":
                return self._measure(tool_input)
            if tool_name == "vmd_represent":
                return self._represent(tool_input)
            return {"ok": False, "output": "", "error": f"unsupported tool {tool_name!r}"}
        except _VmdTimeout as exc:
            self.close()  # restart fresh on the next call
            return {"ok": False, "output": "", "error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "output": "", "error": str(exc)}

    # -------------------------------------------------------------- internals
    def _run_tcl(self, command: str) -> Dict[str, Any]:
        command = command.strip()
        if not command:
            return {"ok": False, "output": "", "error": "empty command"}
        # Reject syntactically incomplete Tcl (unbalanced braces/brackets/quotes) BEFORE
        # sending it — otherwise the interpreter waits for more input and the bridge hangs
        # until timeout (the recurring `{x y z]` / mismatched-bracket failure).
        if not _tcl_is_complete(command):
            return {"ok": False, "output": "",
                    "error": "incomplete Tcl: unbalanced braces/brackets/quotes — not executed. "
                             "Fix the brackets (e.g. `[list $i $j]`, coords as `{x y z}`)."}
        printed, rc, msg = self._exec_wrapped(command, timeout=self._timeout)
        printed = _strip_noise(printed)
        if rc == 0:
            parts = [p for p in (printed.rstrip("\n"), msg.strip()) if p.strip()]
            return {"ok": True, "output": _truncate("\n".join(parts) or "(command ok, no output)"), "error": ""}
        return {"ok": False, "output": _truncate(printed.rstrip("\n")), "error": _truncate(msg.strip()) or "Tcl error"}

    def _snapshot(self, purpose: str, save_path: Optional[str] = None) -> Dict[str, Any]:
        out = os.path.join(tempfile.gettempdir(),
                           f"vmdai_snap_{os.getpid()}_{int(time.time() * 1000)}.tga")
        with contextlib.suppress(Exception):
            if os.path.exists(out):
                os.unlink(out)
        _printed, _rc, msg = self._exec_wrapped(
            "display update\nrender TachyonInternal {" + out + "}", timeout=self._timeout
        )
        result: Dict[str, Any] = {"ok": True, "error": ""}
        if os.path.isfile(out) and os.path.getsize(out) > 1000:
            png = _to_png_bytes(out)
            if png:
                result["image_b64"] = base64.b64encode(png).decode()
                result["image_mime"] = "image/png"   # Anthropic rejects image/x-tga
                note = f"Snapshot rendered ({purpose or 'viewport'})."
                if save_path:
                    saved = _persist_image(png, str(save_path))
                    note += f" Saved to {saved}." if saved else f" (FAILED to save to {save_path})."
                result["output"] = note
            else:
                result["output"] = (f"Snapshot rendered to {out}, but TGA->PNG conversion "
                                    "was unavailable (pip install pillow).")
            with contextlib.suppress(Exception):
                os.unlink(out)
        else:
            result["output"] = "Snapshot attempted; no image produced. " + (msg.strip() or "")
        return result

    # ---- semantic tools: the harness writes the correct Tcl so the model can't ----
    _METRIC_TCL = {
        "rgyr":      ('set __s [atomselect top "{sel}"]; set __v [measure rgyr $__s]; $__s delete', True),
        "sasa":      ('set __s [atomselect top "{sel}"]; set __v [measure sasa 1.4 $__s]; $__s delete', True),
        "natoms":    ('set __v [molinfo top get numatoms]', False),
        "nresidues": ('set __s [atomselect top "{sel}"]; set __v [llength [lsort -unique [$__s get residue]]]; $__s delete', True),
        "ca_dist":   ('set __s [atomselect top "{sel} and name CA"]; set __i [$__s get index]; $__s delete; '
                      'set __v [measure bond [list [lindex $__i 0] [lindex $__i end]]]', True),
        "rmsd_self": ('set __s [atomselect top "{sel}"]; set __v [measure rmsd $__s $__s]; $__s delete', True),
    }

    def _measure(self, ti: Dict[str, Any]) -> Dict[str, Any]:
        metric = str(ti.get("metric") or "").strip()
        sel = str(ti.get("selection") or "protein").strip()
        if metric not in self._METRIC_TCL:
            return {"ok": False, "output": "",
                    "error": f"unknown metric {metric!r}; choose from {sorted(self._METRIC_TCL)}"}
        body, uses_sel = self._METRIC_TCL[metric]
        tcl = (body.format(sel=sel) if uses_sel else body) + '\nputs "VMDAI_VALUE=$__v"'
        res = self._run_tcl(tcl)
        if not res.get("ok"):
            return res
        import re as _re
        val = None
        for line in str(res.get("output") or "").splitlines():
            m = _re.search(r"VMDAI_VALUE=\s*([-+0-9.eE]+)", line)
            if m:
                val = m.group(1)
        if val is None:
            return {"ok": False, "output": res.get("output", ""), "error": "could not read measured value"}
        note = f"{metric}({sel}) = {val}"
        save = ti.get("save_path")
        if save:
            try:
                p = os.path.abspath(str(save))
                os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
                with open(p, "w") as fh:
                    fh.write(str(val) + "\n")
                note += f"  (written to {p})"
            except Exception as exc:  # noqa: BLE001
                note += f"  (FAILED to write {save}: {exc})"
        return {"ok": True, "output": note, "error": "", "value": val}

    _STYLES = {"licorice", "newcartoon", "cartoon", "vdw", "cpk", "lines", "newribbons",
               "ribbons", "tube", "trace", "surf", "quicksurf", "points", "bonds", "dynamicbonds"}
    _COLORS = {"name", "element", "charge", "resname", "restype", "resid", "chain",
               "structure", "beta", "occupancy", "index", "colorid", "mass", "type"}

    def _represent(self, ti: Dict[str, Any]) -> Dict[str, Any]:
        style = str(ti.get("style") or "").strip()
        color = str(ti.get("color") or "").strip()
        sel = str(ti.get("selection") or "protein").strip()
        if not style or style.split()[0].lower() not in self._STYLES:
            return {"ok": False, "output": "", "error": f"unknown style {style!r}"}
        cmds = [f"mol modstyle 0 top {style}", f'mol modselect 0 top "{sel}"']
        if color:
            if color.split()[0].lower() not in self._COLORS:
                return {"ok": False, "output": "", "error": f"unknown color {color!r}"}
            cmds.append(f"mol modcolor 0 top {color}")
        res = self._run_tcl("\n".join(cmds))
        if res.get("ok"):
            res["output"] = f'rep 0: style={style}, color={color or "unchanged"}, selection="{sel}"'
        return res

    def _exec_wrapped(self, command: str, timeout: int) -> Tuple[str, int, str]:
        """Run `command` inside catch{}, return (printed_stdout, rc, result_or_error)."""
        script = (
            "set __vbrc [catch {\n" + command + "\n} __vbmsg]\n"
            f'puts "{_RC} $__vbrc"\n'
            f'puts "{_MB}"\nputs $__vbmsg\nputs "{_ME}"\n'
            f'puts "{_DONE}"\nflush stdout\n'
        )
        lines = self._send_and_read(script, timeout)
        printed: List[str] = []
        msg_lines: List[str] = []
        rc = 0
        seen_rc = False
        in_msg = False
        for raw in lines:
            s = _strip_prompt(raw)
            if _RC in s:
                seen_rc = True
                with contextlib.suppress(Exception):
                    rc = int(s.split(_RC, 1)[1].strip().split()[0])
                continue
            if _MB in s:
                in_msg = True
                continue
            if _ME in s:
                in_msg = False
                continue
            if in_msg:
                msg_lines.append(s)
            elif not seen_rc:
                printed.append(s)
        return "\n".join(printed), rc, "\n".join(msg_lines)

    def _send_and_read(self, script: str, timeout: int) -> List[str]:
        if self._master_fd is None:
            raise _VmdTimeout("VMD pty not open")
        os.write(self._master_fd, script.encode("utf-8"))
        out: List[str] = []
        deadline = time.time() + timeout
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise _VmdTimeout(f"VMD command timed out after {timeout}s")
            ready, _, _ = select.select([self._master_fd], [], [], min(remaining, 1.0))
            if not ready:
                if self._proc is not None and self._proc.poll() is not None:
                    raise _VmdTimeout("VMD process exited unexpectedly")
                continue
            try:
                chunk = os.read(self._master_fd, 65536)
            except OSError:
                raise _VmdTimeout("VMD pty closed (process exited?)")
            if not chunk:
                raise _VmdTimeout("VMD pty reached EOF")
            self._buf += chunk
            while b"\n" in self._buf:
                line, self._buf = self._buf.split(b"\n", 1)
                s = line.decode("utf-8", "replace").rstrip("\r")
                if _DONE in s:
                    return out
                out.append(s)


def _strip_prompt(line: str) -> str:
    s = line
    # VMD prints "vmd >" as its prompt and a bare "?" as the *continuation* prompt
    # for each line of a multi-line (braced) command. With echo disabled these
    # prompts still leak into the stream we read, e.g. "? ? ? ? Info) ...". Strip a
    # leading run of either so the model sees clean output (and doesn't waste its
    # small context, or misread the noise as an error).
    while True:
        t = s.lstrip()
        if t.startswith(_PROMPT):
            s = t[len(_PROMPT):]
        elif t[:1] == "?" and (len(t) == 1 or t[1:2].isspace()):
            s = t[1:]
        else:
            break
    return s


def _to_png_bytes(path: str) -> Optional[bytes]:
    """Convert a rendered TGA to PNG bytes. Anthropic only accepts
    png/jpeg/gif/webp, so raw TGA must be converted before it goes back to the
    model. Prefer the runtime's converter (no hard PIL dep), then Pillow."""
    try:
        from vmd_ai_runtime.image_utils import read_image_as_png_bytes
        data = read_image_as_png_bytes(path)
        if data:
            return data
    except Exception:
        pass
    try:
        import io
        from PIL import Image
        buf = io.BytesIO()
        Image.open(path).save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


def _truncate(s: str) -> str:
    """Cap a tool_result so one verbose VMD dump (file read, $sel get, per-atom
    loop) can't flood the model's context — that text is resent every turn."""
    if len(s) <= _MAX_TOOL_OUTPUT_CHARS:
        return s
    return (s[:_MAX_TOOL_OUTPUT_CHARS]
            + f"\n…[truncated {len(s) - _MAX_TOOL_OUTPUT_CHARS} more chars — do not dump "
              "per-atom/coordinate data or raw files; use selections + summary values]")


def _persist_image(png_bytes: bytes, save_path: str) -> Optional[str]:
    """Write a rendered image to save_path (relative -> CWD = the case working dir).
    Honors the requested extension (.png default; .jpg/.tga/.bmp via Pillow if present)."""
    try:
        dest = os.path.abspath(save_path)
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        ext = os.path.splitext(dest)[1].lower()
        if ext in (".jpg", ".jpeg", ".tga", ".bmp"):
            try:
                import io
                from PIL import Image
                Image.open(io.BytesIO(png_bytes)).convert("RGB").save(dest)
                return dest
            except Exception:
                pass  # fall through to a raw PNG write under whatever name was asked
        with open(dest, "wb") as fh:
            fh.write(png_bytes)
        return dest
    except Exception:
        return None


def _strip_noise(text: str) -> str:
    """Drop VMD's chattiest startup/info lines so the model sees signal."""
    keep = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith("Info) ") and ("Multithreading" in s or "plugin" in s or s.endswith("...")):
            continue
        keep.append(ln)
    return "\n".join(keep)


def _tcl_is_complete(cmd: str) -> bool:
    """True if ``cmd`` is syntactically complete (braces + brackets balanced). Pure-Python,
    no Tcl interpreter — a tkinter Tcl interp is thread-affine and aborts the process when
    torn down from the wrong thread (the agent runs in a worker thread). The brace/bracket
    balance catches the unbalanced-bracket hangs (the recurring `{x y z]` failure), which is
    the whole point of the check."""
    return _balanced(cmd)


def _balanced(s: str) -> bool:
    brace = brack = 0
    esc = False
    for ch in s:
        if esc:
            esc = False
        elif ch == "\\":
            esc = True
        elif ch == "{":
            brace += 1
        elif ch == "}":
            brace -= 1
        elif ch == "[":
            brack += 1
        elif ch == "]":
            brack -= 1
        if brace < 0 or brack < 0:
            return False
    return brace == 0 and brack == 0
