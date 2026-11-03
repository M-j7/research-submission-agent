"""Executable chat CLI. --demo is offline; --live explicitly enables paid APIs."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

from submission_assistant.runtime.chat import ChatService,build_chat_graph
from submission_assistant.runtime.contracts import Request
from submission_assistant.runtime.memory import MemoryManager
async def run(args):
    from submission_assistant.runtime.factory import create_runtime
    temporary = tempfile.TemporaryDirectory(prefix='submission-demo-') if args.demo else None
    bundle = None
    try:
        bundle = await create_runtime(ROOT, mode='demo' if args.demo else 'live',
                                      data_dir=Path(temporary.name) if temporary else None,
                                      no_multi_query=args.no_multi_query)
        req = Request(user_id=args.user_id, conv_id=args.conversation_id,
                      message=args.message or 'IEEE Access摘要要求和发表费用？',
                      **({'turn_id': args.turn_id} if args.turn_id else {}))
        output = await build_chat_graph(bundle.service).ainvoke({'request': req})
        print(json.dumps({'source_mode': 'controlled_offline_demo_not_quality_test' if args.demo else 'live',
                          'request_id': req.turn_id, 'answer': output['answer'],
                          'routing': output.get('routing_decision'),
                          'agent_results': output.get('agent_results', []),
                          'final_result': output.get('final_result')}, ensure_ascii=False, indent=2), flush=True)
    finally:
        if bundle is not None:
            await bundle.aclose()
        if temporary is not None:
            temporary.cleanup()


def main():
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'): stream.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description='投稿咨询助手运行时；demo不联网，live会产生配置服务费用。')
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--demo',action='store_true')
    mode.add_argument('--live',action='store_true')
    parser.add_argument('--message')
    parser.add_argument('--user-id',default='local-user',help='可信本地调用身份；Web接入时必须由认证后端提供')
    parser.add_argument('--conversation-id',default='local-conversation')
    parser.add_argument('--turn-id',help='重试时使用同一ID；完成后返回已保存答案，不重放工具')
    parser.add_argument('--no-multi-query',action='store_true',help='仅原问题双路检索，方便对照')
    args=parser.parse_args()
    if args.live and not args.message: parser.error('--live requires --message')
    try: asyncio.run(run(args)); return 0
    except KeyboardInterrupt: print('用户已取消。',file=sys.stderr); return 130
    except Exception:
        # Remote bodies, absolute paths and credentials never printed by default.
        print('执行未完成：请检查本地配置、服务权限或记忆存储；未输出异常正文。',file=sys.stderr)
        return 1


if __name__=='__main__': raise SystemExit(main())
