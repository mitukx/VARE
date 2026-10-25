#!/usr/bin/env python3
"""Reconstruct model-output, tool-transport, resource, and no-edit status."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs/model-run-v6';PROTO=ROOT/'model-protocol-v6.json';LOCK=ROOT/'model-protocol-v6.lock.json';WS=ROOT/'model-candidate-v6'
def sha(b):return hashlib.sha256(b).hexdigest()
def hf(p):return sha(p.read_bytes())
def write_new(p,obj):
 with p.open('xb') as f:f.write((json.dumps(obj,sort_keys=True,indent=2)+'\n').encode())
def main():
 protocol=json.loads(PROTO.read_text());lock=json.loads(LOCK.read_text());summary=json.loads((RUN/'summary.json').read_text());agent=json.loads((RUN/'agent.json').read_text())
 checks={};bad=[]
 checks['protocol_locked']=hf(PROTO)==lock['protocol_sha256']==summary['protocol_sha256']
 checks['runner_locked']=hf(ROOT/'model_pilot_runner_v6.py')==lock['model_runner_sha256']==summary['runner_sha256']
 checks['model_run_manifest_valid']=True
 m=json.loads((RUN/'manifest.json').read_text());listed=set()
 for row in m['files']:
  p=RUN/row['path'];listed.add(row['path'])
  if not p.is_file() or p.stat().st_size!=row['size_bytes'] or hf(p)!=row['sha256']:checks['model_run_manifest_valid']=False
 actual={p.relative_to(RUN).as_posix() for p in RUN.rglob('*') if p.is_file() and p.name!='manifest.json'}
 checks['model_run_manifest_complete']=listed==actual
 checks['agent_runner_exited_cleanly']=summary['process']['exit_code']==0 and summary['process']['abort_reason'] is None
 checks['offline_wrapper_used']=summary['process']['argv'][:3]==['/usr/bin/sandbox-exec','-p','(version 1) (allow default) (deny network*)']
 before=agent['source_hashes_before'];after=agent['source_hashes_after'];target='mlx/primitives.cpp';base=protocol['source']['baseline_target_sha256']
 checks['source_unchanged']=before==after=={target:base} and summary['candidate_target_sha256']==base
 checks['empty_diff_hash']=summary['candidate_diff_nonempty'] is False and summary['candidate_diff_sha256']==sha(b'')
 event_lines=(RUN/'events.jsonl').read_text().splitlines();prev='0'*64;events=[];chain=True
 for i,line in enumerate(event_lines,1):
  row=json.loads(line);ev=row['event'];canonical=json.dumps(ev,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode();dig=sha(canonical)
  if ev.get('sequence')!=i or row.get('previous_event_sha256')!=prev or row.get('event_sha256')!=dig:chain=False
  prev=dig;events.append(ev)
 types=[x['event_type'] for x in events]
 generations=[x for x in events if x['event_type']=='model_output_completed']
 tool_results=[x for x in events if x['event_type']=='tool_result']
 checks['event_chain_valid']=chain and prev==agent['event_journal_final_event_sha256']
 checks['generation_event_count_matches']=len(generations)==8==summary['model_output_events']
 checks['tool_result_count_matches']=len(tool_results)==8
 checks['all_tool_outputs_complete']=all(t.get('tool_output_status')=='complete_tool_call' for t in agent['transcript'])
 calls=[(t['tool_calls'][0]['name'],t.get('tool_output_status')) for t in agent['transcript'] if t.get('tool_calls')]
 checks['no_edit_tool_call']=not any(name in {'replace_lines','replace_text','apply_patch','write_file'} for name,_ in calls)
 checks['turn_budget_stop']=summary['agent_outcome']=='turn_budget_exhausted' and agent['tool_turns']==8 and len(agent['transcript'])==8
 checks['no_model_grade_by_rule']=not (ROOT/'runs/model-grade-v6').exists()
 samples=[json.loads(x) for x in (RUN/'resources.jsonl').read_text().splitlines()]
 lim=protocol['resources']
 floors=all(x['available_memory_bytes']>=lim['minimum_available_memory_bytes'] and x['free_swap_bytes']>=lim['minimum_free_swap_bytes'] and x['disk_free_bytes']>=lim['minimum_disk_free_bytes'] for x in samples)
 checks['resource_floors']=floors
 checks['process_rss_within_limit']=max(x['process_tree_rss_bytes'] for x in samples)<=lim['max_model_process_tree_rss_bytes']
 checks['swap_growth_within_limit']=max(x['swap_used_bytes'] for x in samples)-json.loads((RUN/'preflight.json').read_text())['swap_used_bytes']<=lim['max_swap_growth_bytes']
 checks['attempt_classification_no_source_edit']=summary['status']=='NO_SOURCE_EDIT' and summary['source_edit_applied'] is False and agent['edit_applied'] is False
 checks['no_training_or_heldout']=summary['training_updates']==0 and summary['heldout_evaluations']==0
 bad=[k for k,v in checks.items() if not v]
 tool_counts={}
 for name,_ in calls:tool_counts[name]=tool_counts.get(name,0)+1
 token_total=sum(x['generated_tokens'] for x in generations)
 results={'audit':'INDEPENDENT_MODEL_PILOT_RUN_V6','checks':checks,'failed_checks':bad,'status':'AUDIT_PASS_NO_SOURCE_EDIT' if not bad else 'AUDIT_NO_GO',
  'model_attempt':{'task_id':summary['task_id'],'task_role':summary['role'],'model':summary['model'],'revision':summary['model_revision'],
    'turns':agent['tool_turns'],'model_output_events':len(generations),'generated_tokens':token_total,'tool_actions':tool_counts,
    'complete_tool_call_rate':sum(x.get('tool_output_status')=='complete_tool_call' for x in agent['transcript'])/len(agent['transcript']),
    'source_edits_applied':int(agent['edit_applied']),'source_diff_nonempty':summary['candidate_diff_nonempty'],
    'stop_reason':agent['stop_reason'],'attempt_seconds':agent['elapsed_seconds'],'outer_process_seconds':summary['process']['elapsed_seconds'],
    'independent_grade_invocations':0,'training_updates':0,'heldout_evaluations':0},
  'resources':{'maximum_sampled_process_tree_rss_bytes':max(x['process_tree_rss_bytes'] for x in samples),
    'mlx_peak_memory_bytes':agent['runtime'].get('mlx_peak_memory_bytes'),'mlx_active_memory_bytes':agent['runtime'].get('mlx_active_memory_bytes'),
    'minimum_available_memory_bytes':min(x['available_memory_bytes'] for x in samples),
    'minimum_free_swap_bytes':min(x['free_swap_bytes'] for x in samples),
    'maximum_swap_growth_bytes':max(x['swap_used_bytes'] for x in samples)-json.loads((RUN/'preflight.json').read_text())['swap_used_bytes'],
    'cgroup_memory':'unavailable on macOS; not claimed'},
  'tool_trace':calls,'event_count':len(events),'final_event_sha256':prev,
  'run_manifest_sha256':hf(RUN/'manifest.json'),'agent_record_sha256':hf(RUN/'agent.json'),
  'candidate_diff_sha256':summary['candidate_diff_sha256'],
  'interpretation':'All eight model tool calls were complete and valid, but the model read consecutive 100-line ranges from the start of a 6,450-line C++ file and exhausted its turn budget before searching for or editing the relevant symbol. This is a single Development interface/inspection-strategy feasibility failure, not an estimate of coding capability.'}
 out=ROOT/'runs/model-run-v6-independent-audit.json';write_new(out,results);print(json.dumps(results,sort_keys=True,indent=2));return 0 if not bad else 2
if __name__=='__main__':raise SystemExit(main())
