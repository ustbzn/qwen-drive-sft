#!/usr/bin/env python3
"""Convert ReCogDrive InternVL-style trajectory/QA JSONL into the ms-swift
`messages` schema for Qwen3-VL SFT.

Input  (one JSON object per line, produced by run_generate_dataset.py):
    {
      "id": 0,
      "image": ["/abs/path/front.jpg"],
      "token": "...",
      "conversations": [
        {"from": "system", "value": "..."},
        {"from": "human",  "value": "<image>\n... prompt ..."},
        {"from": "gpt",    "value": "Here is the planning trajectory [PT, ...]."}
      ]
    }

Output (ms-swift native multimodal schema):
    {
      "messages": [
        {"role": "system",    "content": "..."},
        {"role": "user",      "content": "<image>\n... prompt ..."},
        {"role": "assistant", "content": "Here is the planning trajectory ..."}
      ],
      "images": ["/abs/path/front.jpg"]
    }

ms-swift can read several sharegpt variants directly, but the explicit
`messages`/`images` form is the most robust across versions, so we normalize to
it here. The `<image>` placeholder in the user turn is preserved; ms-swift's
Qwen3-VL template expands it to the model's real vision tokens.

Usage:
    python scripts/training/convert_recogdrive_jsonl_to_swift.py \
        --in  navsim_traj_front_view.jsonl \
        --out recogdrive_qwen3_sft.jsonl
"""
import argparse
import json

ROLE_MAP = {
    "system": "system",
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
}


def convert_line(obj: dict) -> dict:
    messages = []
    for turn in obj.get("conversations", []):
        role = ROLE_MAP.get(turn.get("from"))
        if role is None:
            raise ValueError(f"Unknown conversation role: {turn.get('from')!r}")
        messages.append({"role": role, "content": turn.get("value", "")})

    images = obj.get("image", [])
    if isinstance(images, str):
        images = [images]

    out = {"messages": messages}
    if images:
        out["images"] = images
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", required=True, help="input jsonl")
    parser.add_argument("--out", dest="out_path", required=True, help="output jsonl")
    args = parser.parse_args()

    n = 0
    with open(args.in_path, "r", encoding="utf-8") as fin, \
            open(args.out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            json.dump(convert_line(obj), fout, ensure_ascii=False)
            fout.write("\n")
            n += 1

    print(f"Converted {n} samples -> {args.out_path}")


if __name__ == "__main__":
    main()
