# Project conventions

## Auto-push after commit

Whenever a `git commit` succeeds (in this repo), immediately follow it with
`git push` to the current branch's tracked remote in the **same** Bash call
(e.g. `git commit -m '...' && git push origin <branch>`).

Why: training happens on a different machine. Every commit must reach
`origin/<branch>` so `git pull` over there picks it up — waiting for the
user to type "push" each time costs a round-trip.

When to skip:
- Commit failed (pre-commit hook etc.) — fix the issue first, do NOT push.
- Branch has no upstream (`git push` would error) — set upstream once with
  `git push -u origin <branch>`, then auto-push as normal afterwards.
- User explicitly says "don't push" / "local only" for that commit.
- Force-push (`--force` / `--force-with-lease`) is never auto: always confirm.
