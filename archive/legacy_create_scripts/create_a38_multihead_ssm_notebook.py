#!/usr/bin/env python3
"""Build A38: A33 base with the SelectiveSSM upgraded to v181's multi-head version.

The graft:
  - wliilamsam's LightProtoSSM already has Cross-Attention + SWA (we verified).
  - The single-head SelectiveSSM (one shared (B,C) projection per d_model dim) is the
    core architectural difference vs udaysonawane v181.
  - v181 uses multi-head SelectiveSSM with `params=6` independent (B,C) projections
    and a learnable `head_mix` to combine them, plus a SiLU(z) gate and out_proj.
  - We drop in the multi-head SelectiveSSM class with `params=4` (compromise — full
    `params=6` triples training time on CPU; 4 keeps it ~2x and still gives most of
    the expressivity gain).

The forward signature `(B, T, D) -> (B, T, D)` is unchanged, so LightProtoSSM (which
calls SelectiveSSM via its forward) needs no edits.

Output: references/public_baselines/wliilamsam_onnx_perch_seq_sed_0943/variants/a38_a33_multihead_ssm.ipynb
"""
from __future__ import annotations

import json
from pathlib import Path


BASE_NB = Path(
    "references/public_baselines/wliilamsam_onnx_perch_seq_sed_0943/variants/"
    "a33_konbu_head_mattia_rescue.ipynb"
)
OUT_NB = Path(
    "references/public_baselines/wliilamsam_onnx_perch_seq_sed_0943/variants/"
    "a38_a33_multihead_ssm.ipynb"
)


# Replacement SelectiveSSM class — v181's multi-head version, params=4 by default.
# Forward signature (B, T, D) -> (B, T, D) unchanged so the LightProtoSSM caller is unaffected.
NEW_SELECTIVE_SSM = """\
class SelectiveSSM(nn.Module):
    \"\"\"A38: multi-head Mamba-style SelectiveSSM grafted from udaysonawane v181.

    `params` controls the number of independent (B, C) projection heads. The hidden
    state has shape (B, D, P, N); per-step outputs across heads are mixed by a learnable
    `head_mix` parameter. `params=4` (vs single-head in wliilamsam) ~doubles compute but
    gives meaningfully richer dynamics.
    \"\"\"
    def __init__(self, d_model, d_state=16, d_conv=4, params=4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.params = params

        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.conv1d = nn.Conv1d(
            d_model, d_model, d_conv, padding=d_conv - 1, groups=d_model
        )
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)

        # Multi-head A initialization
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).unsqueeze(0).expand(d_model, params, -1)  # (D, P, N)
        self.A_log = nn.Parameter(torch.log(A))

        self.D = nn.Parameter(torch.ones(d_model))

        self.B_proj = nn.ModuleList([
            nn.Linear(d_model, d_state, bias=False) for _ in range(params)
        ])
        self.C_proj = nn.ModuleList([
            nn.Linear(d_model, d_state, bias=False) for _ in range(params)
        ])
        self.head_mix = nn.Parameter(torch.full((d_model, params), 1.0 / params))
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B_size, T, D = x.shape
        P, N = self.params, self.d_state

        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)

        x_conv = self.conv1d(x_ssm.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_conv = F.silu(x_conv)

        dt = F.softplus(self.dt_proj(x_conv))         # (B, T, D)
        A = -torch.exp(self.A_log)                    # (D, P, N)

        B = torch.stack([proj(x_conv) for proj in self.B_proj], dim=2)  # (B, T, P, N)
        C = torch.stack([proj(x_conv) for proj in self.C_proj], dim=2)  # (B, T, P, N)

        h = torch.zeros(B_size, D, P, N, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            dt_t = dt[:, t, :, None, None]            # (B, D, 1, 1)
            A_expanded = A[None, :, :, :]             # (1, D, P, N)
            dA = torch.exp(A_expanded * dt_t)         # (B, D, P, N)
            dB = dt_t * B[:, t].unsqueeze(1)          # (B, D, P, N)
            u_t = x_conv[:, t, :, None, None]         # (B, D, 1, 1)
            h = h * dA + u_t * dB
            y_heads = (h * C[:, t].unsqueeze(1)).sum(-1)         # (B, D, P)
            y_t = (y_heads * self.head_mix[None, :, :]).sum(-1)  # (B, D)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)                    # (B, T, D)

        y = y + x_conv * self.D[None, None, :]        # skip
        y = y * F.silu(z)                              # gating
        return self.out_proj(y)
"""


# We replace from "class SelectiveSSM(nn.Module):" up to (but not including) the next
# "class ..." line, which is "class LightProtoSSM(nn.Module):" in wliilamsam's cell.
def patch_cell_source(src: str) -> str:
    start_marker = "class SelectiveSSM(nn.Module):"
    end_marker = "\nclass LightProtoSSM(nn.Module):"
    if start_marker not in src or end_marker not in src:
        raise RuntimeError("A38: could not find SelectiveSSM block boundaries in cell")
    pre, _, rest = src.partition(start_marker)
    _, _, post = rest.partition(end_marker)
    new_src = pre + NEW_SELECTIVE_SSM.rstrip() + "\n\n\n" + end_marker.lstrip() + post
    return new_src


def main() -> None:
    if not BASE_NB.exists():
        raise FileNotFoundError(BASE_NB)
    nb = json.loads(BASE_NB.read_text())

    target_idx = None
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        s = "".join(c.get("source", []))
        if "class SelectiveSSM(nn.Module):" in s and "class LightProtoSSM(nn.Module):" in s:
            target_idx = i
            break
    if target_idx is None:
        raise RuntimeError("A38: did not find LightProtoSSM-defining cell to patch")

    cell = nb["cells"][target_idx]
    old_src = "".join(cell.get("source", []))
    new_src = patch_cell_source(old_src)
    cell["source"] = new_src.splitlines(keepends=True)

    OUT_NB.parent.mkdir(parents=True, exist_ok=True)
    OUT_NB.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT_NB} ({len(nb['cells'])} cells)")
    print(f"  patched cell #{target_idx}: {len(old_src)} -> {len(new_src)} chars")


if __name__ == "__main__":
    main()
