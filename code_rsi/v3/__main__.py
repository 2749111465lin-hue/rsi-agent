"""One runtime; preflight never accesses credentials or sends model requests."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser=argparse.ArgumentParser(description="RAG RSI v3 unified runtime")
    subs=parser.add_subparsers(dest="command",required=True)
    smoke=subs.add_parser("offline-demo",help="isolated synthetic demonstration; zero API")
    smoke.add_argument("--out",required=True)
    for name,help_text in (("preflight","verify a frozen calibration plan; zero API"),
                           ("calibrate","execute an explicitly approved frozen plan"),
                           ("grade","local diagnostics after all answers are frozen")):
        cmd=subs.add_parser(name,help=help_text)
        cmd.add_argument("--plan",required=True)
        if name=="calibrate":
            cmd.add_argument("--execute",action="store_true",required=True)
            cmd.add_argument("--approved-plan-hash",required=True)
    args=parser.parse_args(argv)
    project=Path(__file__).resolve().parents[2]
    if args.command=="offline-demo":
        from .offline_demo import demo
        target=Path(args.out).resolve()
        if project not in target.parents:
            raise ValueError("run output must remain under the existing project")
        result=demo(target)
    else:
        from .calibration import load_plan, preflight, generate, grade
        plan=load_plan(args.plan)
        target=Path(plan["output_dir"]).resolve()
        runs=(project/"runs").resolve()
        if runs not in target.parents:
            raise ValueError("calibration output must be a child directory under project runs")
        if args.command=="preflight":
            result=preflight(plan)
        elif args.command=="calibrate":
            generate(plan,approved_plan_hash=args.approved_plan_hash)
            result=grade(plan)
        else:
            result=grade(plan)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result

if __name__=="__main__": main()
