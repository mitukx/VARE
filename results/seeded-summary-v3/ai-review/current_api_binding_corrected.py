import copy,json,sys
sys.path[:0]=['/Users/user/Documents/Codex/2026-10-13/var/work/VARE-eval-stats/src','/Users/user/Documents/Codex/2026-10-13/var/work/VARE-eval-stats']
from reproducers.seeded_summary_fixture_v1 import synthetic_grid
from vare.format_vs_correctness_seeded_summary_v3 import ANALYSIS_SHA256,summarize_seeded_comparison
c,p,rows=synthetic_grid()
out={}
for name,tag,prospective in [('no_row_binding',None,True),('wrong_row_binding','wrong-analysis',True),('non_bool_binding',ANALYSIS_SHA256,'yes'),('properly_bound',ANALYSIS_SHA256,True)]:
    changed=copy.deepcopy(rows)
    if tag is not None:
        for row in changed: row['analysis_sha256']=tag
    try:
        summary=summarize_seeded_comparison(c,p,changed,prospective=prospective)
        out[name]={'summary_returned':True,'analysis_status':summary['analysis_status'],'file_pin_verified':summary['analysis_file_pin_verified']}
        assert name=='properly_bound' and summary['analysis_status']=='prospective_rows_bound' and summary['analysis_file_pin_verified'] is False
    except (ValueError,TypeError) as exc:
        out[name]={'summary_returned':False,'error':type(exc).__name__+': '+str(exc)}
        assert name!='properly_bound'
print(json.dumps(out,indent=2))
