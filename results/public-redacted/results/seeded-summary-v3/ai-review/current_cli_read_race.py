import hashlib,json,sys,tempfile
from pathlib import Path
ROOT=Path('<SESSION_ROOT>/work/VARE-eval-stats')
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from reproducers.seeded_summary_fixture_v1 import synthetic_grid
from scripts import summarize_format_vs_correctness_seeded_v3 as cli
from vare import format_vs_correctness_v2 as legacy
from vare.format_vs_correctness_seeded_summary_v3 import summarize_seeded_comparison
confirm,pack,rows=synthetic_grid()
for row in rows: row.update(backend='mlx',model_id='hand-model',protocol_sha256=legacy.PROTOCOL_SHA256)
cli._protocol=lambda:{'models':{'mlx':{'repo_id':'hand-model'}}}
cli.confirm_eval_tasks=lambda:confirm
cli.pack_eval_tasks=lambda _:pack
with tempfile.TemporaryDirectory(dir='<SESSION_ROOT>/work/reviewer/seed-stats') as temp:
    root=Path(temp); run=root/'run';run.mkdir(); cli.ROOT=root/'source'
    fakepack=cli.ROOT/'benchmarks/heldout/exact_integer_v1/tasks.jsonl';fakepack.parent.mkdir(parents=True);fakepack.write_text('hand fixture')
    progress=run/'progress.jsonl';progress.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    before=progress.read_bytes()
    altered=[dict(r) for r in rows]
    for row in altered:
        if row['kind']=='eval' and row['arm']=='format_sft': row['output']='nope'
    next_blob=''.join(json.dumps(r)+'\n' for r in altered).encode()
    real_read=cli.read_jsonl
    def interleaved_reader(path):
        # Simulate another writer between analyze's byte hash read and JSON read.
        path.write_bytes(next_blob)
        return real_read(path)
    cli.read_jsonl=interleaved_reader
    output=run/'analysis.json';cli.analyze(run,output,retrospective=True)
    result=json.loads(output.read_text())
    old=summarize_seeded_comparison(confirm,pack,rows,prospective=False)
    print(json.dumps({'reported_progress_hash_is_old_blob':result['progress_sha256']==hashlib.sha256(before).hexdigest(),'actual_current_progress_hash_matches_report':result['progress_sha256']==hashlib.sha256(progress.read_bytes()).hexdigest(),'primary_changed_from_old_blob':result['paired_bootstrap']!=old['paired_bootstrap'],'summary_emitted':output.exists()},indent=2))
