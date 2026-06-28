"""Training entry point. CLI flags select stage, lambda values, and target ratio.

See instructions.md (gitignored) for the staged plan. Warm-start Stage 3 from the
Stage-2 codec; add gradient clipping; sweep lambda_task in {0, 0.1, 1, 10}.
"""

from __future__ import annotations


def main():
    raise NotImplementedError("Implement staged training (Stages 1-3).")


if __name__ == "__main__":
    main()
