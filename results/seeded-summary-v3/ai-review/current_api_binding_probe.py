import copy,json,sys
sys.path[:0]=['/Users/user/Documents/Codex/2026-10-13/var/work/VARE-eval-stats/src','/Users/user/Documents/Codex/2026-10-13/var/work/VARE-eval-stats']
from reproducers.seeded_summary_fixture_v1 import synthetic_grid
from vare.format_vs_correctness_seeded_summary_v3 import summarize_seeded_comparison
c,p,rows=synthetic_grid()
out={}
for name,tag in [('no_row_binding',None),('wrong_row_binding','wrong-analysis')]:
    changed=copy.deepcopy(rows)
    if tag is not None:
        for row in changed: row['analysis_sha256']=tag
    summary=summarize_seeded_comparison(c,p,changed,prospective=True)
    out[name]={'analysis_status':summary['analysis_status'],'row_has_matching_bindings':False,'summary_returned':True}
print(json.dumps(out,indent=2))
