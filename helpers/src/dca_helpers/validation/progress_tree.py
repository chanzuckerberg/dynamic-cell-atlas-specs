"""
Live progress-tree display for `runner._validate_with_tree`.

Self-contained Rich renderer: consumes only store-path strings and
ZarrNodeValidationResult objects, with no dependency on validation logic.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable

from rich.live import Live
from rich.text import Text
from rich.tree import Tree

from dca_helpers.validation.discovery import well_of_field
from dca_helpers.validation.result import ZarrNodeValidationResult


def _store_label(path: str) -> str:
    return os.path.basename(path.rstrip("/")) or path


class _TreeDisplay:
    """
    Live tree display: only in-flight stores appear as tree nodes.
    Completed stores are printed above the Live area via live.console.
    The live tree is therefore bounded to max_workers active stores at
    a time regardless of total store count.

    All tree mutations are protected by a single lock. rich.Live renders
    the tree on its own timer; CPython's GIL makes the list operations
    safe against concurrent renders without explicit lock coordination.
    """

    def __init__(self, root_path: str, total_stores: int) -> None:
        self._root_path = root_path
        self._queued = total_stores
        self._active = 0
        self._done = 0
        self.tree = Tree(self._make_root_label())
        self._store_nodes: dict[str, Tree] = {}
        self._lock = threading.Lock()
        self._live: Live | None = None

    def _make_root_label(self) -> Text:
        t = Text(self._root_path, style="bold")
        parts: list[Text] = []
        if self._active:
            parts.append(Text(f"{self._active} active", style="yellow"))
        if self._queued:
            parts.append(Text(f"{self._queued} queued", style="dim"))
        if self._done:
            parts.append(Text(f"{self._done} done", style="green"))
        if parts:
            t.append("  ")
            for i, p in enumerate(parts):
                if i:
                    t.append(" · ", style="dim")
                t.append_text(p)
        return t

    def set_live(self, live: Live) -> None:
        self._live = live

    def start_store(self, store_path: str) -> None:
        """Add a store node to the live tree. Called from worker threads."""
        label = _store_label(store_path)
        with self._lock:
            node = self.tree.add(Text.assemble(("⟳  ", "yellow"), (label, "bold")))
            self._store_nodes[store_path] = node
            self._queued -= 1
            self._active += 1
            self.tree.label = self._make_root_label()

    def make_field_hook(
        self, store_path: str
    ) -> Callable[[list[str]], dict[str, Callable[[], None]]]:
        """
        Return a hook to pass to validate_zarr_node. Called after start_store.
        The hook is invoked with the discovered field paths; it adds well
        sub-nodes and returns a per-field advance callable.
        """
        lock = self._lock

        def hook(
            field_paths: list[str],
        ) -> dict[str, Callable[[], None]]:
            store_node = self._store_nodes.get(store_path)
            if store_node is None:
                return {}

            label = _store_label(store_path)
            well_fields: dict[str, list[str]] = {}
            for fp in field_paths:
                well_fields.setdefault(well_of_field(fp), []).append(fp)

            n_fields = len(field_paths)
            well_state: dict[str, dict] = {}
            total_done = [0]

            with lock:
                store_node.label = Text.assemble(
                    ("⟳  ", "yellow"),
                    (label, "bold"),
                    (f"  0/{n_fields} fields", "dim"),
                )
                for well in sorted(well_fields):
                    n = len(well_fields[well])
                    wn = store_node.add(
                        Text.assemble(("○  ", "dim"), well, (f"  0/{n}", "dim"))
                    )
                    well_state[well] = {"total": n, "done": 0, "node": wn}

            def make_advance(well: str) -> Callable[[], None]:
                def advance() -> None:
                    with lock:
                        state = well_state[well]
                        state["done"] += 1
                        done, total = state["done"], state["total"]
                        wn = state["node"]
                        if done == total:
                            wn.label = Text.assemble(
                                ("✓  ", "green"), well, (f"  {done}/{total}", "dim")
                            )
                        else:
                            wn.label = Text.assemble(
                                ("⟳  ", "yellow"), well, (f"  {done}/{total}", "dim")
                            )
                        total_done[0] += 1
                        store_node.label = Text.assemble(
                            ("⟳  ", "yellow"),
                            (label, "bold"),
                            (f"  {total_done[0]}/{n_fields} fields", "dim"),
                        )

                return advance

            field_advance_fns: dict[str, Callable[[], None]] = {}
            for well, fps in well_fields.items():
                advance = make_advance(well)
                for fp in fps:
                    field_advance_fns[fp] = advance
            return field_advance_fns

        return hook

    def complete_store(
        self,
        store_path: str,
        store_results: list[ZarrNodeValidationResult],
    ) -> None:
        """Remove store from live tree and print a summary line above it."""
        node = self._store_nodes.pop(store_path, None)
        label = _store_label(store_path)
        is_skipped = store_results and all(r.skipped for r in store_results)
        passed = all(r.passed for r in store_results)
        n_errors = sum(len(r.errors) for r in store_results)
        n_warnings = sum(len(r.warnings) for r in store_results)
        n_fields = sum(1 for r in store_results if r.parent_path is not None)

        if is_skipped:
            parts: list[tuple[str, str]] = [
                ("○  ", "dim"),
                (label, "dim"),
                ("  skipped (GEFF)", "dim"),
            ]
        elif passed:
            parts = [("✓  ", "bold green"), (label, "bold")]
            if n_fields:
                parts.append((f"  {n_fields} fields", "dim"))
            if n_warnings:
                parts.append((f"  {n_warnings} warning(s)", "yellow"))
        else:
            parts = [
                ("✗  ", "bold red"),
                (label, "bold red"),
                (f"  {n_errors} error(s)", "red"),
            ]

        with self._lock:
            if node is not None and node in self.tree.children:
                self.tree.children.remove(node)
            self._active -= 1
            self._done += 1
            self.tree.label = self._make_root_label()

        if self._live is not None:
            self._live.console.print(Text.assemble(*parts))
