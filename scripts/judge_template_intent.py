"""Independent BM25/embedding paths; no JEV calls, fusion or routing."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from submission_assistant.intent.bm25 import BM25IntentRecognizer  # noqa: E402
from submission_assistant.intent.embedding import (  # noqa: E402
    EmbeddingIntentRecognizer, EmbeddingSettings, EmbeddingServiceError, BATCH_SIZE,
)
from submission_assistant.intent.templates import load_retrieval_templates, validate_message  # noqa: E402


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="九类业务示例＋非业务对照示例：BM25 / Qwen Embedding 独立匹配，不调用 JEV。")
    parser.add_argument("--message", required=True)
    parser.add_argument("--method", choices=("bm25", "embedding", "both"), default="bm25")
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / ".cache" / "intent_embeddings")
    parser.add_argument("--dry-run", action="store_true", help="展示全部模板及请求规模，不读密钥、不联网")
    args = parser.parse_args()
    output = {}
    try:
        message = validate_message(args.message)
        if args.dry_run:
            templates = load_retrieval_templates()
            count = sum(len(group) for group in templates.values())
            output = {"message": message, "method": args.method, "template_count": count,
                      "templates": templates,
                      "embedding_batch_sizes": [min(BATCH_SIZE, count - offset) for offset in range(0, count, BATCH_SIZE)],
                      "note": f"未联网。缓存未命中时初始化{count}条示例；每次查询另外发送1条消息。"}
        else:
            output = {}
            if args.method in ("bm25", "both"):
                output["bm25"] = BM25IntentRecognizer().recognize(message).to_dict()
            if args.method in ("embedding", "both"):
                settings = EmbeddingSettings.from_env(args.env_file)
                output["embedding"] = asyncio.run(EmbeddingIntentRecognizer(
                    settings, cache_dir=args.cache_dir).recognize(message)).to_dict()
        print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, EmbeddingServiceError) as exc:
        if args.method == "both" and "bm25" in output:
            output["embedding"] = {"status": "error", "error": str(exc)}
            print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
        print(f"模板意图判断失败：{exc}", file=sys.stderr)
        return 1
    except OSError:
        print("读取本地文件失败，请检查路径与访问权限。", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
