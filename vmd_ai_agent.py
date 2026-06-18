"""
vmd_ai_agent.py — SciVisAgentBench adapter for the vmd_ai (ChatVMD) runtime.

Wraps your existing ``ClaudeToolLoop`` so it can be evaluated by
SciVisAgentBench's framework. The loop is driven exactly as in production —
same system prompt, same tool surface — except VMD tool calls are executed by
an in-process headless VMD (see headless_vmd_bridge.py) instead of the Tk panel.

Install: drop this file + headless_vmd_bridge.py into
``benchmark/evaluation_framework/agents/`` and add to that package's
``__init__.py``:

    from .vmd_ai_agent import VmdAiAgent

Then run, e.g.:

    python -m benchmark.evaluation_framework.run_evaluation \
        --agent vmd_ai \
        --config benchmark/configs/vmd_ai/config_anthropic.json \
        --yaml benchmark/eval_cases/molecular_vis/eval_analysis_tasks.yaml \
        --cases SciVisAgentBench-tasks/molecular_vis \
        --eval-model gpt-5.2 --experiment-number exp1
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

# --- make the framework + this dir importable when run as a package member ---
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # for evaluation_framework
sys.path.insert(0, str(Path(__file__).parent))                # for the sibling bridge
from evaluation_framework.base_agent import BaseAgent, AgentResult  # noqa: E402
from evaluation_framework.agent_registry import register_agent       # noqa: E402

# Semantic tools (#3): the harness emits the correct VMD Tcl, so the model can't get the
# error-prone syntax wrong — it just names the operation. Advertised to the loop via
# ClaudeToolLoop.extra_tools and executed by the SubprocessVmdBridge.
VMD_MEASURE_SCHEMA = {
    "name": "vmd_measure",
    "description": ("Compute a numeric analysis value with correct VMD Tcl (you cannot get the "
                    "syntax wrong). Returns the value and can write it to a file."),
    "input_schema": {
        "type": "object",
        "properties": {
            "metric": {"type": "string",
                       "enum": ["rgyr", "sasa", "natoms", "nresidues", "ca_dist", "rmsd_self"],
                       "description": ("rgyr=radius of gyration (Å); sasa=SASA (Å², 1.4 probe); "
                                       "natoms=total atoms; nresidues=protein residue count; "
                                       "ca_dist=distance between first and last CA (Å); rmsd_self=RMSD vs self (0)")},
            "selection": {"type": "string", "description": "VMD atom selection (default 'protein')"},
            "save_path": {"type": "string", "description": "optional absolute path to write the value to"},
        },
        "required": ["metric"],
    },
}
VMD_REPRESENT_SCHEMA = {
    "name": "vmd_represent",
    "description": "Set the molecule's representation (style / coloring / selection) with correct VMD Tcl.",
    "input_schema": {
        "type": "object",
        "properties": {
            "style": {"type": "string", "description": "e.g. Licorice, NewCartoon, VDW, Lines, NewRibbons"},
            "color": {"type": "string", "description": "e.g. Name, Element, Charge, ResName, Structure, 'ColorID 1'"},
            "selection": {"type": "string", "description": "VMD atom selection (default 'protein')"},
        },
        "required": ["style"],
    },
}

# VMD backends live in sibling modules and are imported lazily in setup()
# based on config['vmd_backend'] ('subprocess' = native VMD.app, default;
# 'vmd_python' = in-process vmd-python).


class _NoopQueue:
    """ClaudeToolLoop passes a session_queue down to the tool bridge; the
    headless bridge ignores it, and run() never calls it directly."""

    def push(self, *args, **kwargs):
        return None


def _resolve_provider(config: Dict[str, Any]):
    """Map a SciVis config to ClaudeToolLoop's (provider_name, api_key, model).

    For ollama, ClaudeToolLoop repurposes the api_key slot to carry the base URL.
    """
    provider = str(config.get("provider") or "anthropic").lower()
    model = config.get("model") or "claude-sonnet-4-5"
    if provider in ("anthropic", "anthropic-direct", "anthropic_api"):
        key = config.get("api_key") or os.environ.get("ANTHROPIC_API_KEY", "")
        return "anthropic-direct", key, model
    if provider in ("openrouter", "open-router", "vllm", "local", "openai"):
        key = (config.get("api_key")
               or os.environ.get("OPENROUTER_API_KEY")
               or os.environ.get("OPENAI_API_KEY")
               or os.environ.get("ANTHROPIC_AUTH_TOKEN")
               or "EMPTY")  # local vLLM may need no key
        return "openrouter", key, model
    if provider in ("ollama", "local-ollama", "local_ollama"):
        base = (config.get("base_url")
                or os.environ.get("VMD_AI_OLLAMA_BASE_URL")
                or "http://localhost:11434")
        return "ollama", base, model
    key = config.get("api_key") or os.environ.get("ANTHROPIC_API_KEY", "")
    return "anthropic-direct", key, model


@register_agent("vmd_ai")
class VmdAiAgent(BaseAgent):
    """Evaluates the vmd_ai/ChatVMD agent (ClaudeToolLoop) head­lessly."""

    def __init__(self, config: Dict[str, Any]):
        config.setdefault("agent_name", "vmd_ai")
        config.setdefault("eval_mode", "mcp")  # result-dir naming compatibility
        super().__init__(config)
        self._loop = None
        self._bridge = None
        self._system_prompt = ""

    # ----------------------------------------------------------------- setup
    async def setup(self):
        # Put the vmd_ai runtime on sys.path, then import its loop + prompt.
        runtime_path = (self.config.get("vmd_ai_runtime_path")
                        or os.environ.get("VMD_AI_RUNTIME_PATH"))
        if not runtime_path:
            raise RuntimeError(
                "Set 'vmd_ai_runtime_path' in the config (or VMD_AI_RUNTIME_PATH) "
                "to <your repo>/vmd_ai/runtime so 'vmd_ai_runtime' is importable."
            )
        if runtime_path not in sys.path:
            sys.path.insert(0, runtime_path)

        from vmd_ai_runtime.claude_loop import ClaudeToolLoop, VMD_SYSTEM_PROMPT

        provider_name, api_key, model = _resolve_provider(self.config)
        # Point the OpenAI-style path at a custom endpoint (local vLLM/SGLang) if given.
        base_url = self.config.get("base_url") or os.environ.get("VMD_AI_OPENAI_BASE_URL")
        if base_url:
            os.environ["VMD_AI_OPENAI_BASE_URL"] = base_url
            print(f"[vmd_ai] OpenAI-compatible endpoint: {base_url}")
        if provider_name in ("anthropic-direct", "openrouter") and not api_key:
            raise RuntimeError(
                f"No API key for provider {provider_name!r}. Set ANTHROPIC_API_KEY / "
                "OPENROUTER_API_KEY, or put 'api_key' in the config."
            )
        # --- optional retrieval augmentations (the A/B arms) -----------------
        # Mirrors RuntimeApp (runtime/app.py): RAG adds a `docs_search` tool over
        # the VMD/Tcl reference; Wiki adds the `wiki_list/read/update` tools.
        # Toggled per arm via config so the same cases run under none/rag/wiki/both.
        docs_search = None
        if bool(self.config.get("enable_rag", False)):
            from vmd_ai_runtime.docs_search import DocsSearch
            docs_search = DocsSearch(index_dir=self.config.get("docs_index_dir"))
            avail = bool(getattr(docs_search, "is_available", False))
            print(f"[vmd_ai] RAG on: docs_search.is_available={avail}"
                  + ("" if avail else "  (no index -> run 'vmd-ai-index --rebuild'; this arm is a NO-OP)"))
        wiki_store = None
        if bool(self.config.get("enable_wiki", False)):
            from vmd_ai_runtime.wiki_store import WikiStore
            home = os.path.expanduser("~")
            # env override lets a sweep give each (arm, seed) a fresh wiki root.
            w_root = (os.environ.get("VMD_AI_WIKI_ROOT")
                      or self.config.get("wiki_root") or os.path.join(home, ".vmdai", "wiki"))
            r_root = self.config.get("wiki_raw_root") or os.path.join(home, ".vmdai", "raw")
            try:
                wiki_store = WikiStore(w_root, raw_root=r_root)
                wiki_store.bootstrap()
                print(f"[vmd_ai] Wiki on: {w_root}")
            except Exception as exc:  # noqa: BLE001
                print(f"[vmd_ai] Wiki bootstrap failed ({exc}); this arm runs without wiki")
                wiki_store = None

        self._loop = ClaudeToolLoop(
            provider_name=provider_name,
            api_key=api_key,
            model=model,
            timeout=int(self.config.get("loop_timeout", 120)),
            docs_search=docs_search,
            wiki_store=wiki_store,
        )
        self._system_prompt = (
            VMD_SYSTEM_PROMPT
            + "\n\nMode: agent (SciVisAgentBench)."
            + "\n\nDELIVERABLES: when a task asks you to SAVE a screenshot/image/render or a "
              "VMD state to a specific path, you MUST create that exact file. For images, call "
              "capture_vmd_snapshot with save_path set to the requested path (it writes a real "
              "PNG/JPG). For a .vmd state, run_vmd_command \"save_state {path}\". For text answers, "
              "write via run_vmd_command (set f [open {path} w]; puts $f ...; close $f). Substitute "
              "your agent_mode into any {agent_mode} placeholder. capture_vmd_snapshot WITHOUT "
              "save_path only lets you inspect the view and saves nothing."
            + "\n\nEFFICIENCY: do NOT read or print raw structure files (no open/read of the .cif/.pdb), "
              "full coordinate lists, or per-atom data — it floods context, burns tokens, and is resent "
              "every turn. Load with 'mol new <file>' (VMD parses it; you never need its raw contents) and "
              "inspect via atom selections and summary values (counts, a single measure), not bulk dumps. "
              "Be concise and finish in as few turns as possible."
        )
        # retrieve-first arm: the model has a docs_search tool but won't use it
        # spontaneously; this directive forces a lookup before risky commands.
        if bool(self.config.get("retrieve_first", False)) and docs_search is not None:
            self._system_prompt += (
                "\n\nRETRIEVAL (mandatory): a docs_search tool over the VMD/Tcl reference is available. "
                "BEFORE you call run_vmd_command with any VMD command whose exact argument syntax you are "
                "not 100% certain of — especially measure (bond / dihed / angle / contacts / rgyr / rmsf), "
                "atomselect, and mol — FIRST call docs_search for that command and follow the returned "
                "syntax exactly. One quick docs_search beats guessing and getting a Tcl error."
            )

        # context-injection arm: put a reference doc directly in the system prompt
        # (no retrieval, no tool call) — tests whether the knowledge helps when it is
        # GUARANTEED present, separating "won't retrieve" from "can't use it".
        ref_path = self.config.get("inject_reference_path")
        if ref_path:
            try:
                with open(os.path.expanduser(ref_path)) as fh:
                    ref = fh.read()
                self._system_prompt += "\n\nVMD TCL REFERENCE — use these exact, correct idioms:\n" + ref
                print(f"[vmd_ai] injected reference: {ref_path} ({len(ref)} chars)")
            except Exception as exc:  # noqa: BLE001
                print(f"[vmd_ai] could not inject reference {ref_path}: {exc}")

        backend = str(self.config.get("vmd_backend", "subprocess")).lower()
        if backend in ("subprocess", "vmd_app", "vmdapp", "native"):
            from subprocess_vmd_bridge import SubprocessVmdBridge
            self._bridge = SubprocessVmdBridge(
                vmd_bin=self.config.get("vmd_bin"),
                timeout=int(self.config.get("vmd_timeout", 120)),
            )
        else:  # 'vmd_python'
            from headless_vmd_bridge import HeadlessVmdBridge
            self._bridge = HeadlessVmdBridge()
        # auto-RAG-on-error: harness-side retrieval that augments a Tcl error with the
        # correct syntax from the docs index (no docs_search tool exposed to the model).
        # Tests whether action-triggered retrieval gets the inject-level win automatically.
        if bool(self.config.get("auto_rag_on_error", False)):
            from vmd_ai_runtime.docs_search import DocsSearch
            from retrieval_bridge import RetrievalAugmentingBridge
            ds = DocsSearch(index_dir=self.config.get("docs_index_dir"))
            self._bridge = RetrievalAugmentingBridge(self._bridge, docs_search=ds)
            print(f"[vmd_ai] auto-RAG-on-error: docs.is_available={getattr(ds, 'is_available', False)}")

        # smart-inject: task-conditioned PROACTIVE injection. Unlike auto_rag_on_error
        # (which fires after a failure), retrieve the reference relevant to the task and
        # inject it BEFORE the model acts — inject's timing with autorag's scoping. The
        # retrieval itself runs per-task in run_task(); here we just build the handle.
        self._smart_docs = None
        if bool(self.config.get("smart_inject", False)):
            from vmd_ai_runtime.docs_search import DocsSearch
            self._smart_docs = DocsSearch(index_dir=self.config.get("docs_index_dir"))
            print(f"[vmd_ai] smart-inject: docs.is_available={getattr(self._smart_docs, 'is_available', False)}")

        # semantic tools (#3): advertise vmd_measure / vmd_represent (executed by the bridge),
        # which emit correct Tcl so the model never writes the error-prone measure/atomselect syntax.
        if bool(self.config.get("enable_semantic_tools", False)):
            self._loop.extra_tools = [VMD_MEASURE_SCHEMA, VMD_REPRESENT_SCHEMA]
            self._system_prompt += (
                "\n\nPREFER TOOLS: for a numeric value call vmd_measure(metric, selection[, save_path]) — it runs "
                "the correct VMD Tcl and returns the number. For a representation call vmd_represent(style, color, "
                "selection). Use run_vmd_command only for what these don't cover."
            )
            print("[vmd_ai] semantic tools enabled: vmd_measure, vmd_represent")

        print(f"[vmd_ai] ready: provider={provider_name} model={model} backend={backend}")

    async def teardown(self):
        if self._bridge is not None and hasattr(self._bridge, "close"):
            import contextlib
            with contextlib.suppress(Exception):
                self._bridge.close()

    @staticmethod
    def _write_tcl_transcript(tcl_log, out_dir, case_name, task_description=""):
        """Write the agent's run_vmd_command calls as a runnable .tcl script: commands that
        ran without a Tcl error become the working script; errored attempts are listed as
        comments. Lets every solved task emit a reusable VMD script artifact."""
        if not tcl_log:
            return None
        ok_cmds = [e["cmd"].rstrip() for e in tcl_log if e.get("ok")]
        fails = [e for e in tcl_log if e.get("ok") is False]
        first = (task_description or "").strip().splitlines()
        lines = [
            f"# vmd_ai working Tcl transcript — {case_name}",
            f"# task: {first[0][:100] if first else ''}",
            "# Commands below ran without a Tcl error, in order.  Reproduce with:",
            "#   vmd -dispdev text -e <this file>",
            "",
        ] + ok_cmds
        if fails:
            lines += ["", "# --- attempts that errored (excluded from the script above) ---"]
            for e in fails:
                c = (e["cmd"].strip().splitlines() or [""])[0][:90]
                er = (e.get("err") or "").strip().splitlines()
                lines.append(f"#   {c}    ;# ERR: {er[0][:90] if er else ''}")
        try:
            path = os.path.join(out_dir, f"{case_name}.tcl")
            with open(path, "w") as fh:
                fh.write("\n".join(lines) + "\n")
            return path
        except Exception:
            return None

    # ------------------------------------------------------------------ task
    async def run_task(self, task_description: str, task_config: Dict[str, Any]) -> AgentResult:
        if self._loop is None:
            await self.setup()

        start = time.time()
        timeout_s = int(task_config.get("timeout", self.config.get("task_timeout", 900)))
        working_dir = (task_config.get("working_dir")
                       or task_config.get("data_dir")
                       or os.getcwd())
        prev_cwd = os.getcwd()
        collected: list[str] = []
        cancel_event = __import__("threading").Event()

        def on_chunk(text: str) -> None:
            collected.append(text)

        # Visible tool I/O for debugging: set VMD_AI_DEBUG_TOOLS=1 to print every
        # run_vmd_command (the exact Tcl) + its result. Off by default so real runs
        # stay quiet. This is the window into *why* a weak model stalls.
        _debug_tools = os.environ.get("VMD_AI_DEBUG_TOOLS", "").lower() in ("1", "true", "yes", "on")
        tcl_log: list = []   # one entry per run_vmd_command -> the .tcl transcript

        def _short(s: Any, n: int = 600) -> str:
            s = str(s)
            return s if len(s) <= n else s[:n] + f" …(+{len(s) - n} chars)"

        def on_tool_start(name: str, tool_input: Dict[str, Any]) -> None:
            if name == "run_vmd_command":
                tcl_log.append({"cmd": str(tool_input.get("command", "")), "ok": None, "err": ""})
            if not _debug_tools:
                return
            if name == "run_vmd_command":
                print("\n  → run_vmd_command:\n      "
                      + _short(tool_input.get("command", "")).replace("\n", "\n      "),
                      flush=True)
            else:
                print(f"\n  → {name}: {_short(tool_input)}", flush=True)

        def on_tool_result(tool_id: str, name: str, result: Dict[str, Any]) -> None:
            if name == "run_vmd_command" and tcl_log:
                tcl_log[-1]["ok"] = bool(result.get("ok"))
                tcl_log[-1]["err"] = str(result.get("error") or "")
            if not _debug_tools:
                return
            tag = "ok" if result.get("ok") else "ERR"
            line = f"  ← {name} [{tag}]"
            if result.get("error"):
                line += f" error={_short(result.get('error'), 300)}"
            if result.get("output"):
                line += f" output={_short(result.get('output'), 300)}"
            print(line, flush=True)

        # smart-inject: retrieve the reference relevant to THIS task and inject up front.
        system_prompt = self._system_prompt
        if getattr(self, "_smart_docs", None) is not None and getattr(self._smart_docs, "is_available", False):
            try:
                # scope to the curated reference (vmd_ref) only — retrieving from "all"
                # pulls in long skill-workflow chunks that dilute the precise syntax.
                r = self._smart_docs.search(task_description, k=2, scope="vmd_ref")
                chunks = [str(c.get("text") or "").strip() for c in (r.get("results") or [])]
                ref = "\n\n---\n".join(t for t in chunks if t)[:2500]
                if ref:
                    system_prompt = (self._system_prompt
                        + "\n\nVMD TCL REFERENCE (retrieved for this task — use these exact idioms):\n" + ref)
                    if _debug_tools:
                        print(f"  [smart-inject] added {len(ref)} chars from {len(chunks)} chunks (scope=vmd_ref)", flush=True)
            except Exception:
                pass

        def do_run() -> str:
            return self._loop.run(
                prompt=task_description,
                system_prompt=system_prompt,
                tool_bridge=self._bridge,
                session_id="scivis",
                session_queue=_NoopQueue(),
                cancel_event=cancel_event,
                on_chunk=on_chunk,
                on_tool_start=on_tool_start,
                on_tool_result=on_tool_result,
                prior_messages=None,
            )

        try:
            os.chdir(working_dir)            # so data/<file> + results/ paths resolve
            self._bridge.reset()            # fresh VMD scene per case
            loop = asyncio.get_running_loop()
            final_text = await asyncio.wait_for(
                loop.run_in_executor(None, do_run), timeout=timeout_s
            )
        except asyncio.TimeoutError:
            cancel_event.set()
            return AgentResult(
                success=False,
                error=f"task timed out after {timeout_s}s",
                metadata={"duration": time.time() - start, "timeout": True,
                          "partial_response": "".join(collected) or None},
            )
        except Exception as exc:
            return AgentResult(
                success=False,
                error=f"{exc}\n\n{traceback.format_exc()}",
                metadata={"duration": time.time() - start},
            )
        finally:
            os.chdir(prev_cwd)
            self._write_tcl_transcript(tcl_log, working_dir,
                                       str(task_config.get("case_name") or "task"),
                                       task_description)

        response = final_text or "".join(collected)
        duration = time.time() - start

        # ClaudeToolLoop does not surface provider token counts, so estimate.
        input_tokens = self.count_tokens(task_description) + 1500  # ~sys prompt + tool schemas
        output_tokens = self.count_tokens(response)

        tcl_path = os.path.join(working_dir, f"{task_config.get('case_name') or 'task'}.tcl")
        tcl_path = tcl_path if os.path.exists(tcl_path) else None
        dirs = self.get_result_directories(task_config["case_dir"], task_config["case_name"])
        return AgentResult(
            success=True,
            response=response,
            output_files={"results_dir": str(dirs["results_dir"]), "tcl_script": tcl_path},
            metadata={
                "duration": duration,
                "assistant_response": response,
                "tcl_script": tcl_path,
                "tcl_ok_commands": sum(1 for e in tcl_log if e.get("ok")),
                "_token_info": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "source": "estimated",
                },
            },
        )
