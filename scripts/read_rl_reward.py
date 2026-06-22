#!/usr/bin/env python3
"""Read RL training scalars (reward + Plan D diagnostics) from TensorBoard events.

The RL launcher redirects stdout to a .txt where `reward=` is a tqdm progress-bar
field clobbered by carriage returns -- not reliably greppable. The reliable source
is the TensorBoard event file. This prints the per-epoch series for reward and the
flow-GRPO diagnostics so you can watch a run and confirm the Step 1+2 change took.

Examples:
  # newest run of the default flow-RL experiment
  python scripts/read_rl_reward.py

  # a specific experiment name / explicit exp root
  python scripts/read_rl_reward.py --name training_recogdrive_qwen3_flow_RL \
      --exp-root /pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/0312_0319/exp

  # a specific run timestamp instead of the newest
  python scripts/read_rl_reward.py --run 2026.06.17.11.08.39

  # only the last 6 points
  python scripts/read_rl_reward.py --tail 6

  # point straight at an event file or its directory
  python scripts/read_rl_reward.py --event /path/to/lightning_logs/version_0
"""
import argparse
import glob
import os
import sys

DEFAULT_EXP_ROOT = os.environ.get(
    "NAVSIM_EXP_ROOT",
    "/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/0312_0319/exp",
)
DEFAULT_NAME = "training_recogdrive_qwen3_flow_RL"
DEFAULT_TAGS = [
    "train/reward_epoch",
    "train/reward_std_epoch",
    "train/flow_std_epoch",
    "train/logprob_clamp_hit_rate_epoch",
    "train/logprob_mean_epoch",
    "train/advantage_std_epoch",
    "train/policy_loss_epoch",
    "train/bc_loss_epoch",
    "train/loss_epoch",
]


def find_event(args):
    if args.event:
        p = args.event
        if os.path.isdir(p):
            hits = sorted(
                glob.glob(os.path.join(p, "**", "events.out.tfevents.*"), recursive=True),
                key=os.path.getmtime,
            )
            if not hits:
                sys.exit(f"no event file under {p}")
            return hits[-1]
        return p
    pattern = os.path.join(
        args.exp_root, args.name, "*", "lightning_logs", "version_0",
        "events.out.tfevents.*",
    )
    hits = sorted(glob.glob(pattern), key=os.path.getmtime)
    if args.run:
        hits = [h for h in hits if args.run in h]
    if not hits:
        sys.exit(
            f"no event files matched: {pattern}"
            + (f" (run filter: {args.run})" if args.run else "")
        )
    return hits[-1]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--exp-root", default=DEFAULT_EXP_ROOT)
    ap.add_argument("--name", default=DEFAULT_NAME)
    ap.add_argument("--run", default=None, help="substring of a run timestamp dir; default newest")
    ap.add_argument("--event", default=None, help="explicit event file or dir (overrides --name/--run)")
    ap.add_argument("--tags", nargs="*", default=DEFAULT_TAGS)
    ap.add_argument("--tail", type=int, default=0, help="only show the last N points (0 = all)")
    args = ap.parse_args()

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ev = find_event(args)
    print("reading:", ev)
    acc = EventAccumulator(ev, size_guidance={"scalars": 0})
    acc.Reload()
    available = set(acc.Tags()["scalars"])

    for tag in args.tags:
        if tag not in available:
            print(f"{tag:38s} (missing)")
            continue
        vals = [s.value for s in acc.Scalars(tag)]
        shown = vals[-args.tail:] if args.tail > 0 else vals
        series = ", ".join(f"{v:.4f}" for v in shown)
        print(f"{tag:38s} [{len(vals)}] {series}")

    # quick sanity hints for the Step 1+2 sampling change
    if "train/flow_std_epoch" in available:
        last_std = acc.Scalars("train/flow_std_epoch")[-1].value
        if abs(last_std - 0.05) < 1e-4:
            verdict = "OK (Step 1+2 active: flow_std lowered to 0.05)"
        elif abs(last_std - 0.1) < 1e-4:
            verdict = "still 0.1 -> running OLD code (Step 1+2 not in effect)"
        else:
            verdict = "unexpected value -- check FlowConfig"
        print(f"\nflow_std last = {last_std:.4f}  -> {verdict}")
    if "train/reward_epoch" in available:
        r = [s.value for s in acc.Scalars("train/reward_epoch")]
        print(
            f"reward first={r[0]:.4f} last={r[-1]:.4f}  "
            f"(pre-fix baseline last=0.4781; aim toward DDIM rollout reward ~0.677)"
        )


if __name__ == "__main__":
    main()
