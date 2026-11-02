import copy,hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path('<SESSION_ROOT>/work/VARE-eval-stats')
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from reproducers.seeded_summary_fixture_v1 import synthetic_grid
from scripts import summarize_format_vs_correctness_seeded_v3 as cli
from scripts import run_format_vs_correctness_comparison_v2 as driver
from vare import format_vs_correctness_v2 as legacy
from vare.format_vs_correctness_seeded_summary_v3 import ANALYSIS_SHA256,bind_analysis
confirm,pack,original=synthetic_grid()
def forbidden(*a,**kw): raise AssertionError('real confirm generator/model session must not be used')
legacy.confirm_eval_tasks=forbidden
cli._protocol=lambda:{'models':{'mlx':{'repo_id':'hand-synthetic-model'}}}
cli.confirm_eval_tasks=lambda:copy.deepcopy(confirm)
cli.pack_eval_tasks=lambda _:copy.deepcopy(pack)
driver._Mlx=driver._Transformers=forbidden
old_module=(ROOT/'src/vare/format_vs_correctness_v2.py').read_bytes()
old_protocol=(ROOT/'protocols/format_vs_correctness_comparison_v2.json').read_bytes()
results={}
with tempfile.TemporaryDirectory(dir='<SESSION_ROOT>/work/reviewer/seed-stats') as tmp:
    scratch=Path(tmp); cli.ROOT=scratch/'fake-source'
    packfile=cli.ROOT/'benchmarks/heldout/exact_integer_v1/tasks.jsonl'
    packfile.parent.mkdir(parents=True); packfile.write_text('handmade fixture only')
    for case in ['retrospective','prospective','missing_seed','wrong_row_tag','duplicate_task_id']:
        run=scratch/case; run.mkdir()
        rows=copy.deepcopy(original)
        prospective=case!='retrospective'
        if prospective: bind_analysis(run)
        for row in rows:
            row.update(backend='mlx',model_id='hand-synthetic-model',protocol_sha256=legacy.PROTOCOL_SHA256)
            if prospective: row['analysis_sha256']=ANALYSIS_SHA256
        if case=='missing_seed': rows=[r for r in rows if r['run_seed']!=legacy.RUN_SEEDS[0]]
        if case=='wrong_row_tag': rows[-1]['analysis_sha256']='wrong'
        cli.confirm_eval_tasks=(lambda:copy.deepcopy(confirm)+[copy.deepcopy(confirm[0])]) if case=='duplicate_task_id' else (lambda:copy.deepcopy(confirm))
        progress=run/'progress.jsonl'; progress.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        legacy_report=run/'summary.json'; legacy_report.write_bytes(b'unchanged legacy sentinel\n')
        before=progress.read_bytes(); output=run/'seeded.json'
        error=None
        try: cli.analyze(run,output,retrospective=not prospective)
        except (ValueError,SystemExit) as exc: error=type(exc).__name__+': '+str(exc)
        results[case]={'rejected':error is not None,'error':error,'output_exists':output.exists(),'raw_unchanged':progress.read_bytes()==before,'legacy_report_unchanged':legacy_report.read_bytes()==b'unchanged legacy sentinel\n'}
        if output.exists():
            summary=json.loads(output.read_text()); results[case]['analysis_status']=summary['analysis_status']
            try: cli.analyze(run,output,retrospective=not prospective)
            except FileExistsError: results[case]['existing_output_overwrite_rejected']=True
        assert (error is not None)==(case in ['missing_seed','wrong_row_tag','duplicate_task_id'])
        assert results[case]['raw_unchanged'] and results[case]['legacy_report_unchanged']
assert (ROOT/'src/vare/format_vs_correctness_v2.py').read_bytes()==old_module
assert (ROOT/'protocols/format_vs_correctness_comparison_v2.json').read_bytes()==old_protocol
print(json.dumps({'cases':results,'legacy_module_and_protocol_unchanged':True,'real_confirm_generator_called':False,'model_session_constructed':False},indent=2))
