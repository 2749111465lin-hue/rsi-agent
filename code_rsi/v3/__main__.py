"""One entry point for the new runtime. Live runs require an explicit frozen plan."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description="RAG RSI v3 unified runtime")
    subs=parser.add_subparsers(dest="command",required=True)
    smoke=subs.add_parser("offline-demo",help="real isolated code execution, synthetic model, zero API")
    smoke.add_argument("--out",required=True)
    args=parser.parse_args()
    target=Path(args.out).resolve()
    project=Path(__file__).resolve().parents[2]
    if project not in target.parents:
        raise ValueError("run output must remain under the existing project")
    if args.command=="offline-demo":
        from .offline_demo import demo
        print(json.dumps(demo(target),ensure_ascii=False,indent=2))

if __name__=="__main__": main()
