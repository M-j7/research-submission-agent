"""Local single-worker web launcher. No paid calls until a live message is sent."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description='投稿咨询助手本地网页；demo离线，live发送问题会产生模型费用。')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--demo', action='store_true')
    mode.add_argument('--live', action='store_true')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error('端口须为1024至65535')
    import uvicorn
    from submission_assistant.web.app import create_app
    uvicorn.run(create_app(ROOT, mode='demo' if args.demo else 'live', port=args.port),
                host='127.0.0.1', port=args.port, workers=1, access_log=False)


if __name__ == '__main__':
    main()
