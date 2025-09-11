#!/usr/bin/env python3
"""
usage: python prune_ckpt.py /path/to/dir [mod_base=50] [--apply]
默认 dry-run，只打印；加 --apply 才会真正删除。
"""

import re
import sys
import os
from pathlib import Path
import argparse
import shutil

def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir", nargs="?", type=Path, default="checkpoints/prm/Qwen2.5-1.5B-Instruct-numina-grpo-prm-eta0")
    p.add_argument("mod_base", nargs="?", type=int, default=50)
    p.add_argument("--apply", action="store_true", help="实际删除（默认仅打印）")
    args = p.parse_args()

    d = args.dir
    if not d.is_dir():
      print(f"Not a directory: {d}", file=sys.stderr)
      sys.exit(1)

    # 匹配 global_step_123 或 global_step-123
    pat = re.compile(r"^(global_step[-_]?)(\d+)(.*)$")

    # 收集所有出现过的 step
    steps = set()
    # 只看包含 ckpt 的文件名（覆盖 .ckpt、.ckpt.index、.ckpt.data-* 等）
    for f in d.iterdir():
        if not f.is_dir():
            continue
        name = f.name
        # if "ckpt" not in name:
        #     continue
        m = pat.match(name)
        if m:
            steps.add(int(m.group(2)))

    # 对于非倍数的 step，删除同前缀的所有文件
    to_delete = []
    for step in sorted(steps):
        if step % args.mod_base != 0:
            prefix1 = f"global_step_{step}"
            prefix2 = f"global_step-{step}"
            for f in d.iterdir():
                if not f.is_dir():
                    continue
                n = f.name
                if n.startswith(prefix1) or n.startswith(prefix2):
                    to_delete.append(f)

    if not args.apply:
        print("[Dry-run] 将删除以下文件：")
        for f in to_delete:
            print(f"rm -rf {f}")
        print("\n实际删除请加 --apply")
    else:
        for f in to_delete:
            try:
                shutil.rmtree(f)
            except FileNotFoundError:
                pass
        print(f"已删除 {len(to_delete)} 个文件。")

if __name__ == "__main__":
    main()
