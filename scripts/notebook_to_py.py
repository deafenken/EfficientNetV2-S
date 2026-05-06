#!/usr/bin/env python3
import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: notebook_to_py.py input.ipynb output.py")

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    nb = json.loads(input_path.read_text(encoding="utf-8"))

    output = []
    summaries = []
    code_idx = 0
    for cell in nb.get("cells", []):
        source = "".join(cell.get("source", []))
        if cell.get("cell_type") == "code":
            code_idx += 1
            output.append(f"# %% cell {code_idx}\n{source}\n")
            first_line = source.splitlines()[0] if source.splitlines() else ""
            summaries.append((str(code_idx), len(source.splitlines()), first_line[:120]))
        else:
            first = " ".join(source.strip().split())[:120]
            summaries.append(("md", len(source.splitlines()), first))

    output_path.write_text("\n".join(output), encoding="utf-8")
    print(f"cells={len(nb.get('cells', []))} code_cells={code_idx}")
    for kind, n_lines, first in summaries:
        print(f"{kind:>3} {n_lines:>4} {first}")


if __name__ == "__main__":
    main()

