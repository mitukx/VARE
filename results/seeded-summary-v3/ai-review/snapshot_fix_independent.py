import contextlib,copy,hashlib,io,json,sys,tempfile
from pathlib import Path
ROOT=Path('/Users/user/Documents/Codex/2026-10-13/var/work/VARE-eval-stats')
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from reproducers.seeded_summary_fixture_v1 import synthetic_grid
from scripts import summarize_format_vs_correctness_seeded_v3 as cli
from scripts import run_format_vs_correctness_comparison_v2 as driver
from vare import format_vs_correctness_v2 as legacy
from vare.format_vs_correctness_seeded_summary_v3 import ANALYSIS_SHA256,bind_analysis,read_progress_snapshot,summarize_seeded_comparison
confirm,pack,fixture_rows=synthetic_grid()
def forbidden(*args,**kwargs): raise AssertionError('actual confirmation/model code must not execute')
legacy.confirm_eval_tasks=forbidden
cli.confirm_eval_tasks=driver.confirm_eval_tasks=lambda:copy.deepcopy(confirm)
cli.pack_eval_tasks=driver.pack_eval_tasks=lambda _:copy.deepcopy(pack)
protocol={'models':{'mlx':{'repo_id':'hand-synthetic-model'}}}
cli._protocol=driver._protocol=lambda:protocol
driver._environment=lambda backend:{'backend':backend,'execution':'synthetic-no-model'}
class FakeSession:
    def __init__(self,*a,**kw): pass
    train=prepare_eval=generate=forbidden
driver._Mlx=driver._Transformers=FakeSession
rows=copy.deepcopy(fixture_rows)
for row in rows: row.update(backend='mlx',model_id='hand-synthetic-model',protocol_sha256=legacy.PROTOCOL_SHA256,analysis_sha256=ANALYSIS_SHA256)
blob=''.join(json.dumps(row)+'\n' for row in rows).encode()
altered=copy.deepcopy(rows)
for row in altered:
    if row['kind']=='eval' and row['arm']=='format_sft': row['output']='no answer'
changed=''.join(json.dumps(row)+'\n' for row in altered).encode()
expected_old=legacy.summarize_comparison(confirm,pack,rows)
assert expected_old['paired_bootstrap']!=legacy.summarize_comparison(confirm,pack,altered)['paired_bootstrap']
real_read=Path.read_bytes
out={'cli':[],'driver':None,'parsing':[]}
with tempfile.TemporaryDirectory(dir='/Users/user/Documents/Codex/2026-10-13/var/work/reviewer/seed-stats') as temp:
    root=Path(temp);fakepack=root/'fake-source/benchmarks/heldout/exact_integer_v1/tasks.jsonl'
    fakepack.parent.mkdir(parents=True);fakepack.write_text('handmade fixture only')
    cli.ROOT=root/'fake-source';driver.PACK_PATH=fakepack
    for prospective in [False,True]:
        run=root/f'cli-{prospective}';run.mkdir()
        if prospective: bind_analysis(run)
        progress=run/'progress.jsonl';progress.write_bytes(blob)
        oldfile=run/'summary.json';oldfile.write_text('legacy unchanged sentinel')
        count=[0]
        def capture_and_replace(path):
            got=real_read(path)
            if path==progress:
                count[0]+=1;path.write_bytes(changed)
            return got
        Path.read_bytes=capture_and_replace
        try: cli.analyze(run,run/'seeded.json',retrospective=not prospective)
        finally: Path.read_bytes=real_read
        report=json.loads((run/'seeded.json').read_text())
        expected=summarize_seeded_comparison(confirm,pack,rows,prospective=prospective)
        checks={'prospective':prospective,'one_progress_capture':count[0]==1,'hash_matches_captured_bytes':report['progress_sha256']==hashlib.sha256(blob).hexdigest(),'metrics_match_captured_rows':report['paired_bootstrap']==expected['paired_bootstrap'] and report['curves']==expected['curves'],'writer_replacement_preserved':real_read(progress)==changed,'legacy_summary_unchanged':oldfile.read_text()=='legacy unchanged sentinel','analysis_status':report['analysis_status'],'file_pin_verified':report['analysis_file_pin_verified']}
        assert all(checks[k] for k in ['one_progress_capture','hash_matches_captured_bytes','metrics_match_captured_rows','writer_replacement_preserved','legacy_summary_unchanged'])
        assert report['analysis_status']==('prospective_rows_bound' if prospective else 'retrospective_analysis')
        assert report['analysis_file_pin_verified'] is prospective
        out['cli'].append(checks)
    run=root/'driver';run.mkdir();bind_analysis(run);progress=run/'progress.jsonl';progress.write_bytes(blob)
    count=[0]
    def driver_capture_and_replace(path):
        got=real_read(path)
        if path==progress:
            count[0]+=1;path.write_bytes(changed)
        return got
    Path.read_bytes=driver_capture_and_replace
    try:
        with contextlib.redirect_stdout(io.StringIO()): result=driver.execute('mlx',run)
    finally: Path.read_bytes=real_read
    old=json.loads((run/'summary.json').read_text());new=json.loads((run/'summary.seeded-v3.json').read_text())
    checks={'exit_zero':result==0,'one_final_progress_capture':count[0]==1,'legacy_matches_captured_rows':old==expected_old,'seeded_primary_matches_legacy':new['paired_bootstrap']==old['paired_bootstrap'],'seeded_hash_matches_captured_bytes':new['progress_sha256']==hashlib.sha256(blob).hexdigest(),'writer_replacement_preserved':real_read(progress)==changed,'prospective_row_bound':new['analysis_status']=='prospective_rows_bound','file_pin_verified':new['analysis_file_pin_verified']}
    assert all(checks.values());out['driver']=checks
    for name,data in [('truncated_json',b'{"kind":'),('partial_tail',blob+b'{"kind":'),('nonobject',b'[]\n'),('invalid_utf8',b'\xff'),('complete_without_newline',blob.rstrip(b'\n')),('blank_line_prefix',b'\n \n'+blob)]:
        path=root/f'{name}.jsonl';path.write_bytes(data)
        error=None
        try: got,digest=read_progress_snapshot(path)
        except ValueError as exc: error=type(exc).__name__
        accepted=name in ['complete_without_newline','blank_line_prefix']
        assert (error is None)==accepted
        checks={'case':name,'rejected':error is not None,'error_type':error}
        if accepted:
            assert got==rows and digest==hashlib.sha256(data).hexdigest()
            checks.update(parsed_rows_exact=True,digest_matches_exact_blob=True)
        out['parsing'].append(checks)
out.update(actual_confirm_generator_called=False,actual_model_session_called=False,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'src/vare/format_vs_correctness_seeded_summary_v3.py',ROOT/'scripts/summarize_format_vs_correctness_seeded_v3.py',ROOT/'scripts/run_format_vs_correctness_comparison_v2.py']})
print(json.dumps(out,indent=2))
