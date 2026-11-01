import argparse, importlib.util, json, pathlib, sys
p=argparse.ArgumentParser(); p.add_argument('--task',required=True); p.add_argument('--source',required=True); p.add_argument('--public',action='store_true'); a=p.parse_args()
path=pathlib.Path(a.source)
spec=importlib.util.spec_from_file_location('candidate_retry_policy',path)
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
ok=all(module.retry_budget(n)==n for n in (0,1,5,32))
print(json.dumps({'success':ok,'checks':4,'task':a.task}))
sys.exit(0 if ok else 1)
