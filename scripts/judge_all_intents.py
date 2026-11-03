"""Complete pre-routing scores, with a cost-free preview mode."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from submission_assistant.intent.ensemble import EnsembleIntentRecognizer, WEIGHTS  # noqa: E402
from submission_assistant.intent.jev import JevIntentRecognizer, JevSettings, build_requests  # noqa: E402
from submission_assistant.intent.embedding import EmbeddingIntentRecognizer, EmbeddingSettings, BATCH_SIZE  # noqa: E402
from submission_assistant.intent.templates import load_retrieval_templates  # noqa: E402


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="JEV + Embedding + BM25 三路意图评分，停在路由之前；真实运行会消耗服务余额。")
    parser.add_argument("--message", required=True)
    parser.add_argument("--history-file", type=Path)
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / ".cache" / "intent_embeddings")
    parser.add_argument("--dry-run", action="store_true", help="展示请求结构和评分规则，不读密钥、不联网")
    args = parser.parse_args()
    try:
        try:
            history = json.loads(args.history_file.read_text(encoding="utf-8-sig")) if args.history_file else None
        except (OSError, ValueError):
            raise ValueError("读取历史失败，请检查 UTF-8 JSON 消息数组。") from None
        if args.dry_run:
            preview = build_requests(args.message, history, provider='decisions')
            count = sum(len(group) for group in load_retrieval_templates().values())
            output = {"jev_requests": preview, "template_count": count,
                      "embedding_batch_sizes": [min(BATCH_SIZE, count - offset) for offset in range(0, count, BATCH_SIZE)],
                      "weights": dict(WEIGHTS), "normalization": "九类业务正分占比加权仅为诊断，不改变JEV独立原分，不用于路由。",
                      "note": "JEV十项独立相关性与clarity分为6/5两批；Embedding/BM25只提供辅助证据。不调用RAG。"}
        else:
            recognizer = EnsembleIntentRecognizer(
                JevIntentRecognizer(JevSettings.from_env(args.env_file)),
                EmbeddingIntentRecognizer(EmbeddingSettings.from_env(args.env_file), cache_dir=args.cache_dir))
            output = asyncio.run(recognizer.recognize(args.message, history))
        print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
        return 1 if output.get("degraded") else 0
    except ValueError as exc:
        print(f"三路意图评分失败：{exc}", file=sys.stderr)
        return 1
    except OSError:
        print("本地文件读取失败，请检查路径与权限。", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
