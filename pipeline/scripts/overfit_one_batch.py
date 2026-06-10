"""Overfit one batch (killer gate #1) — run and eyeball the decoded captions."""

from __future__ import annotations

from capit.overfit import run_overfit


def main() -> None:
    result = run_overfit()
    curve = result["ce_curve"]
    for s in range(0, len(curve), max(1, len(curve) // 10)):
        print(f"step {s:4d}  ce {curve[s]:.4f}")
    print(f"final ce: {result['final_ce']:.4f}\n")
    for got, want in zip(result["decoded"], result["targets"]):
        print(f"  target: {' '.join(want)}")
        print(f"  greedy: {' '.join(got)}\n")


if __name__ == "__main__":
    main()
