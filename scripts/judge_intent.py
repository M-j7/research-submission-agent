"""Run with --dry-run to inspect the payload without credentials or API cost."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from submission_assistant.intent import (  # noqa: E402
    IntentInputError, JevConfigurationError, JevIntentRecognizer,
    JevProtocolError, JevServiceError, JevSettings, build_requests,
)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="JEV 十类独立相关性 + 需求清晰度；两批请求，真实运行消耗服务余额。")
    parser.add_argument("--message", required=True)
    parser.add_argument("--history-file", type=Path, help="UTF-8 JSON role/content 数组，最近五轮完整 user/assistant 问答")
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument("--dry-run", action="store_true", help="只展示请求体，不读密钥、不联网")
    args = parser.parse_args()
    try:
        history = json.loads(args.history_file.read_text(encoding="utf-8-sig")) if args.history_file else None
        if args.dry_run:
            output = {'jev_requests': build_requests(args.message, history, provider='decisions'), 'batch_sizes': [6, 5],
                      'score_type': 'independent_relevance_probability'}
        else:
            settings = JevSettings.from_env(args.env_file)
            output = asyncio.run(JevIntentRecognizer(settings).recognize(args.message, history)).to_dict()
        print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
        return 0 if args.dry_run or output['complete'] else 1
    except (IntentInputError, JevConfigurationError, JevProtocolError, JevServiceError) as exc:
        print(f"意图判断失败：{exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError):
        print("读取历史失败：请检查文件路径与 UTF-8 JSON 格式。", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
