"""One frozen, local-7B collector integration preflight; no learner update.

RVL owns the agent loop, rollout engine and lease. This experiment supplies a
local MLX inference/transport backend and an evaluator-owned resettable grader.
Full-sequence reconstruction is an approximation to cached decoding; the
predeclared reconstruction gate must pass before any repaired group is admitted.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def hidden_cases():
    # Frozen exhaustive balanced words, not sampled or filtered using model output.
    cases = []
    for n in range(1, 6):
        for word in itertools.product("()", repeat=2*n):
            s = "".join(word)
            depth, maximum, valid = 0, 0, True
            for c in s:
                depth += 1 if c == "(" else -1
                maximum = max(maximum, depth)
                valid &= depth >= 0
            if valid and depth == 0:
                cases.append([s, [maximum]])
    return cases


GRADER_CHILD = '''
import ast,json,sys,resource
resource.setrlimit(resource.RLIMIT_CPU,(1,2))
data=json.load(sys.stdin)
tree=ast.parse(data['source'])
for node in ast.walk(tree):
 if isinstance(node,(ast.Import,ast.ImportFrom)):
  if not isinstance(node,ast.ImportFrom) or node.module!='typing': raise ValueError('import disallowed')
 if isinstance(node,ast.Name) and node.id.startswith('__'): raise ValueError('private name')
 if isinstance(node,ast.Attribute) and node.attr not in ['append','split','strip','extend','count','join']: raise ValueError('attribute disallowed')
 if isinstance(node,(ast.ClassDef,ast.With,ast.AsyncWith)): raise ValueError('construct disallowed')
tree.body=[n for n in tree.body if not isinstance(n,ast.ImportFrom)]
safe={k:globals()['__builtins__'].__dict__[k] for k in ['len','range','enumerate','zip','max','min','sum','abs','list','tuple','int','str','bool','reversed']}
ns={'__builtins__':safe,'List':list}
exec(compile(tree,'candidate.py','exec'),ns)
f=ns['parse_nested_parens']
print(json.dumps({'passed':all(f(x)==y for x,y in data['cases'])}))
'''


class Grader:
    def __init__(self, name, cases, events):
        self.name, self.cases, self.events = name, cases, events
        self.artifact = sha(canonical({"name":name,"cases":cases,"runner":GRADER_CHILD}).encode())

    async def __call__(self, generation):
        start = time.perf_counter()
        profile = '(version 1)(allow default)(deny network*)(deny file-write*)(deny file-read* (subpath "/Users/user"))'
        args = ['/usr/bin/sandbox-exec','-p',profile,
                '/opt/homebrew/Caskroom/miniforge/base/bin/python3.12','-I','-c',GRADER_CHILD]
        try:
            p = subprocess.run(args, input=canonical({"source":generation.response,"cases":self.cases}),
                               text=True,capture_output=True,timeout=3)
            passed = p.returncode == 0 and json.loads(p.stdout).get('passed') is True
            detail = {"returncode":p.returncode,"stdout":p.stdout,"stderr":p.stderr[-1000:]}
        except (subprocess.TimeoutExpired, ValueError) as exc:
            passed, detail = False, {"error":type(exc).__name__}
        self.events.append({"phase":"grader","feedback":self.name,"artifact_sha256":self.artifact,
                            "state_sha256":sha(generation.response.encode()),"elapsed_s":time.perf_counter()-start,
                            "reward":float(passed),**detail})
        return float(passed)


class LocalKernel:
    def __init__(self, model, tokenizer, protocol, events):
        import mlx.core as mx
        import numpy as np
        from mlx_lm.models.cache import make_prompt_cache
        self.mx,self.np,self.cache = mx,np,make_prompt_cache
        self.model,self.tokenizer,self.protocol,self.events = model,tokenizer,protocol,events
        self.eos = set(tokenizer.eos_token_ids)

    def encode(self, prompt):
        return list(self.tokenizer.apply_chat_template([{"role":"user","content":prompt}],
                   add_generation_prompt=True,tokenize=True))

    def distribution(self, logits, temperature):
        x = self.np.asarray(logits,dtype=self.np.float64)/temperature
        x -= x.max(axis=-1,keepdims=True)
        q = self.np.exp(x)
        return q/q.sum(axis=-1,keepdims=True)

    def decode(self, prompt_ids, temperature, seed, phase, initial_ids=None, initial_logps=None):
        mx,np = self.mx,self.np
        ids,logps = list(initial_ids or []),list(initial_logps or [])
        if ids and ids[-1] in self.eos:
            return ids,logps
        start = time.perf_counter()
        cache = self.cache(self.model)
        context = prompt_ids + ids
        logits = self.model(mx.array([context]),cache=cache)[:,-1,:].astype(mx.float32)
        mx.eval(logits)
        prefill = time.perf_counter()-start
        rng = np.random.default_rng(seed)
        sampled = 0
        while len(ids) < self.protocol['max_new_tokens']:
            p = self.distribution(np.array(logits)[0],temperature)
            token = int(rng.choice(len(p),p=p))
            ids.append(token);logps.append(float(np.log(p[token])));sampled += 1
            if token in self.eos or len(ids) >= self.protocol['max_new_tokens']:
                break
            logits = self.model(mx.array([[token]]),cache=cache)[:,-1,:].astype(mx.float32)
            mx.eval(logits)
        elapsed = time.perf_counter()-start
        self.events.append({'phase':phase,'operation':'generation','elapsed_s':elapsed,
                            'prefill_s':prefill,'context_tokens':len(context),'sampled_tokens':sampled})
        return ids,logps

    def reconstruct(self, prompt_ids, response_ids, q_temperature, p_temperature):
        mx,np = self.mx,self.np
        start = time.perf_counter()
        tokens = prompt_ids + response_ids[:-1]
        logits = self.model(mx.array([tokens]))[0,len(prompt_ids)-1:,:].astype(mx.float32)
        mx.eval(logits)
        raw = np.array(logits)
        q,p = self.distribution(raw,q_temperature),self.distribution(raw,p_temperature)
        self.events.append({'phase':'repair','operation':'q_p_reconstruction_same_weights',
                            'elapsed_s':time.perf_counter()-start,'tokens':len(tokens),
                            'vocab_size':raw.shape[-1],'distribution_host_bytes':q.nbytes+p.nbytes})
        return q,p

    def generation(self, prompt_id, prompt, ids, logps, temperature, elapsed, extra=None):
        from rvl_systems.types import Generation
        return Generation(prompt_id,prompt,self.tokenizer.decode(ids,skip_special_tokens=True),
            sum(logps),len(ids),elapsed,{'backend':'experimental-mlx-local','model':self.protocol['model'],
            'model_revision':self.protocol['model_revision'],'sampling_temperature':temperature,
            'prompt_token_ids':self.encode(prompt),'response_token_ids':ids,
            'response_token_logprobs':logps,'logprob_distribution':'float64_softmax_of_mlx_float32_logits',
            **(extra or {})})


class Backend:
    def __init__(self, kernel, phase, donor=None, acceptance_seed=None):
        self.kernel,self.phase,self.donor = kernel,phase,donor
        self.turn = 0
        self.invalidated = False
        self.diagnostics = []
        self.accept_rng = kernel.np.random.default_rng(acceptance_seed)

    async def generate(self, prompt_id, prompt, *, n, temperature, seed):
        assert n == 1
        k,np = self.kernel,self.kernel.np
        start = time.perf_counter()
        prompt_ids = k.encode(prompt)
        extra = {'state_snapshot_sha256':sha(prompt.encode()),'donor_id':None}
        old = (self.donor.generations[self.turn] if self.donor is not None and
               self.turn<len(self.donor.generations) and not self.invalidated else None)
        if old is not None and old.prompt != prompt:
            self.invalidated = True
            self.diagnostics.append({'turn':self.turn,'reason':'changed_history_drop_all_suffix',
                                     'old_prompt_sha256':sha(old.prompt.encode()),'new_prompt_sha256':sha(prompt.encode())})
            old = None
        if old is None:
            ids,logps = k.decode(prompt_ids,temperature,seed,self.phase)
        else:
            old_ids = old.metadata['response_token_ids']
            assert old.metadata['prompt_token_ids'] == prompt_ids
            q,p = k.reconstruct(prompt_ids,old_ids,old.metadata['sampling_temperature'],temperature)
            reconstructed = np.log(q[np.arange(len(old_ids)),old_ids])
            discrepancy = float(np.max(np.abs(reconstructed-np.array(old.metadata['response_token_logprobs']))))
            # Numerical identity check for the standard residual coupling kernel.
            mass = np.minimum(q,p)
            residual = np.maximum(p-q,0)
            oracle = mass + residual
            law_error = float(np.max(np.abs(oracle-p)))
            diag = {'turn':self.turn,'sampled_logprob_reconstruction_max_abs_error':discrepancy,
                    'numeric_coupling_max_abs_probability_error':law_error,
                    'mean_token_tv':float(np.abs(p-q).sum(axis=-1).mean()/2),
                    'donor_token_count':len(old_ids)}
            self.diagnostics.append(diag)
            if discrepancy > k.protocol['conformance']['sampled_logprob_reconstruction_max_abs_error']:
                raise RuntimeError('frozen sampled-logprob reconstruction gate failed')
            if law_error > k.protocol['conformance']['numeric_coupling_max_abs_probability_error']:
                raise RuntimeError('frozen numerical coupling gate failed')
            ids,logps = [],[]
            for i,token in enumerate(old_ids):
                if self.accept_rng.random() < min(1.,p[i,token]/q[i,token]):
                    ids.append(token);logps.append(float(np.log(p[i,token])))
                    continue
                residual = np.maximum(p[i]-q[i],0)
                assert residual.sum()>0
                token = int(np.random.default_rng(seed).choice(len(residual),p=residual/residual.sum()))
                ids.append(token);logps.append(float(np.log(p[i,token])))
                self.invalidated = True
                diag['rejected_at_token'] = i
                # Different seed stream for the residual draw and its continuation.
                ids,logps = k.decode(prompt_ids,temperature,seed+1000000,self.phase,ids,logps)
                break
            diag['accepted_prefix_tokens'] = diag.get('rejected_at_token',len(old_ids))
            extra['donor_id'] = self.donor.episode_id
            extra['transport_note'] = 'finite-vector coupling with gated numerical reconstruction; not proof of exact backend-law identity'
        self.turn += 1
        return [k.generation(prompt_id,prompt,ids,logps,temperature,time.perf_counter()-start,extra)]


async def experiment(args, protocol, result):
    from mlx_lm import load
    import mlx.core as mx
    from rvl_systems.rollout import AsyncRolloutEngine
    from rvl_systems.lab.serving import ServingLease
    from rvl_systems.lab.tool_agent import CodingToolAgent
    from rvl_systems.types import Generation
    class KernelLease(ServingLease):
        async def generate(self, request):
            rows=await super().generate(replace(request,temperature=self.temperature))
            return [replace(g,metadata={**g.metadata,'feedback_artifact_sha256':self.feedback_sha,
                 'policy_artifact_sha256':self.policy_sha}) for g in rows]

    task_json=json.loads((args.protocol_dir/'human-eval-task.json').read_text())
    task=SimpleNamespace(task_id='preflight-v2/'+task_json['task_id'],prompt=(
        'Implement the complete Python function specified below, including its signature. '
        'Use JSON tools only: edit the implementation, call public_test, then finish. '
        'Do not use markdown or explanatory text.\n'+task_json['prompt']))
    public_cases=[['(()()) ((())) () ((())()())',[2,3,1,3]],
                  ['() (()) ((())) (((())))',[1,2,3,4]],['(()(())((())))',[4]]]
    events=result['events']
    weak=Grader('weak',public_cases[:1],events)
    strong=Grader('strong',public_cases,events)
    hidden=Grader('hidden',hidden_cases(),events)
    write(args.out/'hidden-cases.json',hidden.cases)
    # Validate the evaluator before collecting any model output.
    fixed=task_json['prompt']+task_json['canonical_solution']
    for name,source,expect in [('reference',fixed,1.),('broken','def parse_nested_parens(s): return [0]',0.)]:
        for grader in [weak,strong,hidden]:
            got=await grader(Generation('grader-'+name,'',source,0,0,0))
            if got != expect: raise RuntimeError('grader harness failed: '+grader.name+'/'+name)
    model_path=Path('/Users/user/.cache/huggingface/hub/models--mlx-community--Mistral-7B-Instruct-v0.3-4bit/snapshots')/protocol['model_revision']
    start=time.perf_counter()
    model,tokenizer=load(str(model_path));model.eval()
    mx.eval(model.parameters())
    result['model_load_s']=time.perf_counter()-start
    result['model_artifact_sha256']=file_sha(model_path/'model.safetensors')
    result['device']=str(mx.default_device())
    kernel=LocalKernel(model,tokenizer,protocol,events)
    donors=[]; backends=[]
    # Target kernels are fixed before donors: no learner, curriculum or success-selection ancestry.
    receipt={'group_id':'preassigned-group-0','member_ids':['donor-0','donor-1'],
             'target_policy_training_inputs':[],'target_feedback_training_inputs':[],
             'target_definition_sha256':sha(canonical({'model_sha':result['model_artifact_sha256'],
                'temperature':protocol['target_temperature'],'feedback_sha':strong.artifact}).encode()),
             'consumed_by_learner':False,'selected_by_reward_length_or_completion':False,
             'scope':'fixed-target diagnostic, not a production ancestry enforcement service'}
    write(args.out/'donor-receipt.json',receipt)
    result['donor_receipt']=receipt
    for phase in protocol['ordering']:
        for member in range(protocol['group_size']):
            donor=donors[member] if phase=='repair' else None
            backend=Backend(kernel,phase,donor,protocol['acceptance_seeds'][member])
            backends.append(backend)
            lease=KernelLease(1 if phase=='donors' else 2,AsyncRolloutEngine(backend,max_concurrency=1))
            lease.temperature=protocol['donor_temperature'] if phase=='donors' else protocol['target_temperature']
            feedback=weak if phase=='donors' else strong
            lease.feedback_sha=feedback.artifact
            lease.policy_sha=sha(canonical({'weights':result['model_artifact_sha256'],
                                           'temperature':lease.temperature}).encode())
            agent=CodingToolAgent(feedback,max_steps=protocol['max_steps'],deadline_s=protocol['max_runtime_s'])
            identity=f'{phase}-{member}'
            result['current_episode']=identity
            episode_start=time.perf_counter()
            try:
                episode=await agent.run(task,lease,identity,seed=protocol[f'{"donor" if phase=="donors" else phase}_seeds'][member],journal=args.out/(identity+'-journal.json'))
            finally:
                result['diagnostics'][identity]=backend.diagnostics
            result['episode_wall_s'][identity]=time.perf_counter()-episode_start
            if phase=='donors': donors.append(episode)
            result['episodes'][identity]=asdict(episode)
            result['hidden_rewards'][identity]=await hidden(Generation(identity,'',episode.source,0,0,0))
            write(args.out/'raw.json',result)
    result['all_complete_groups_returned']=all(f'{phase}-{i}' in result['episodes'] for phase in protocol['ordering'] for i in range(2))
    result['collector_conformance_completed']=True
    result['peak_metal_memory_bytes']=mx.get_peak_memory()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--rvl',type=Path,required=True)
    parser.add_argument('--protocol-dir',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(args.rvl/'src'))
    protocol=json.loads((args.protocol_dir/'protocol.json').read_text())
    assert sha((args.protocol_dir/'protocol.json').read_bytes()) == (args.protocol_dir/'protocol.sha256').read_text().split()[0]
    assert sha((args.protocol_dir/'human-eval-task.json').read_bytes()) == protocol['task_sha256']
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.rvl,text=True).strip()
    assert actual==protocol['rvl_commit'],actual
    assert subprocess.check_output(['git','status','--porcelain'],cwd=args.rvl,text=True)==''
    result={'protocol_sha256':sha((args.protocol_dir/'protocol.json').read_bytes()),
            'runner_sha256':sha(Path(__file__).read_bytes()),'rvl_commit':actual,
            'events':[],'episodes':{},'episode_wall_s':{},'hidden_rewards':{},'diagnostics':{},
            'collector_conformance_completed':False,'paid_spend_usd':0,
            'packages':{p:importlib.metadata.version(p) for p in ['mlx','mlx-lm','numpy','transformers']}}
    start=time.perf_counter()
    def deadline(signum,frame): raise TimeoutError('frozen 600-second total runtime cap')
    signal.signal(signal.SIGALRM,deadline);signal.alarm(protocol['max_runtime_s'])
    try:
        asyncio.run(experiment(args,protocol,result))
        result['status']='completed'
    except Exception as exc:
        result['status']='failed';result['error']=str(exc);result['traceback']=traceback.format_exc()
    finally:
        signal.alarm(0)
        result['total_wall_s']=time.perf_counter()-start
        write(args.out/'raw.json',result)
    print(json.dumps({k:result.get(k) for k in ['status','error','collector_conformance_completed','total_wall_s']},indent=2))
    return 0 if result['status']=='completed' else 1


if __name__=='__main__':
    raise SystemExit(main())
