"""Subprocess fixture implementing the parse worker CLI contract."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--file-id", required=True)
    parser.add_argument("--file-name", required=True)
    args = parser.parse_args()
    mode = os.getenv("FAKE_PARSE_MODE", "success")

    if mode == "fail":
        print("fake parser failure", file=sys.stderr)
        return 7
    if mode == "malformed":
        args.output.write_text("{", encoding="utf-8")
        return 0
    if mode == "wait":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        child = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(300)",
            ]
        )
        args.output.with_suffix(".pids").write_text(
            f"{os.getpid()} {child.pid}", encoding="utf-8"
        )
        while True:
            time.sleep(1)

    payload = {
        "chunks": [
            {
                "file_id": args.file_id,
                "file_name": args.file_name,
                "chunk_id": 0,
                "refs": ["p. 1"],
                "text": "Nội dung từ fake worker.",
            }
        ]
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
