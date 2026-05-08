#!/usr/bin/env python3
import json
import sys
from pathlib import Path


VARIANTS = {
    "a17_two_pass_test_tta": {
        "title": "A17: two-pass SSM with test-time ProtoSSM TTA",
        "replacements": [
            (
                """proto_model.eval()\nwith torch.no_grad():\n    proto_out = proto_model(\n        torch.tensor(emb_te_f, dtype=torch.float32),\n        torch.tensor(sc_te_f,  dtype=torch.float32),\n        site_ids=torch.tensor(test_site_ids, dtype=torch.long),\n        hours   =torch.tensor(test_hour_ids, dtype=torch.long),\n    ).numpy()\nproto_scores_flat = proto_out.reshape(-1, N_CLASSES).astype(np.float32)\n""",
                """proto_out = run_tta_proto(\n    proto_model, emb_te_f, sc_te_f,\n    site_t=torch.tensor(test_site_ids, dtype=torch.long),\n    hour_t=torch.tensor(test_hour_ids, dtype=torch.long),\n    shifts=[0, 1, -1, 2, -2],\n)\nproto_scores_flat = proto_out.reshape(-1, N_CLASSES).astype(np.float32)\n""",
            ),
        ],
    },
}


def main():
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: create_two_pass_ssm_variants.py source.ipynb output_dir"
        )

    input_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    base_nb = json.loads(input_path.read_text(encoding="utf-8"))
    for name, spec in VARIANTS.items():
        nb = json.loads(json.dumps(base_nb))
        if nb.get("cells") and nb["cells"][0].get("cell_type") == "markdown":
            nb["cells"][0]["source"] += "\n\n" + spec["title"] + "\n"

        replacement_hits = {old: 0 for old, _ in spec["replacements"]}
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            src = "".join(cell.get("source", []))
            for old, new in spec["replacements"]:
                hits = src.count(old)
                if hits:
                    src = src.replace(old, new)
                    replacement_hits[old] += hits
            cell["source"] = src.splitlines(keepends=True)

        missing = [old for old, hits in replacement_hits.items() if hits == 0]
        if missing:
            raise RuntimeError(f"{name}: replacement text not found: {missing}")

        out_path = output_dir / f"{name}.ipynb"
        out_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
