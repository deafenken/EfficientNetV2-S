#!/usr/bin/env python3
"""Convert a jupytext-style ``# %%`` py file to .ipynb (stdlib only).

Cell delimiters (case-sensitive, line must start with the marker):
  ``# %% [markdown]`` — markdown cell (subsequent ``# `` lines stripped).
  ``# %%`` (anything else after) — code cell, copied verbatim.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_cells(text: str) -> list[dict]:
    cells: list[dict] = []
    current: dict | None = None
    for line in text.splitlines(keepends=True):
        stripped = line.rstrip("\n")
        if stripped.startswith("# %% [markdown]"):
            if current is not None:
                cells.append(current)
            current = {"cell_type": "markdown", "source": []}
            continue
        if stripped.startswith("# %%"):
            if current is not None:
                cells.append(current)
            current = {"cell_type": "code", "source": []}
            continue
        if current is None:
            current = {"cell_type": "code", "source": []}
        if current["cell_type"] == "markdown":
            # Strip the leading "# " comment marker that jupytext adds.
            if line.startswith("# "):
                current["source"].append(line[2:])
            elif line.startswith("#"):
                current["source"].append(line[1:].lstrip(" "))
            else:
                current["source"].append(line)
        else:
            current["source"].append(line)
    if current is not None:
        cells.append(current)
    # Strip trailing blank-line tail in each cell so nbformat doesn't render extra space.
    for cell in cells:
        while cell["source"] and cell["source"][-1].strip() == "":
            cell["source"].pop()
    return cells


def cells_to_nb(cells: list[dict]) -> dict:
    nb_cells = []
    for cell in cells:
        c = {
            "cell_type": cell["cell_type"],
            "metadata": {},
            "source": cell["source"],
        }
        if cell["cell_type"] == "code":
            c["execution_count"] = None
            c["outputs"] = []
        nb_cells.append(c)
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
                "mimetype": "text/x-python",
                "file_extension": ".py",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("src", type=Path)
    p.add_argument("dst", type=Path)
    args = p.parse_args()
    text = args.src.read_text()
    cells = parse_cells(text)
    nb = cells_to_nb(cells)
    args.dst.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"wrote {args.dst} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
