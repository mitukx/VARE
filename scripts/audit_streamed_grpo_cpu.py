"""Offline replay of the existing GRPO equation against streamed model artifacts."""
from __future__ import annotations
import argparse
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
from types import ModuleType
from unittest.mock import patch
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[1]
PINNED_VARE_SOURCES={
    "2b33abfd6ba0faed03e0efc3cce1310090777f6c":(
        ROOT/'benchmarks/streamed_grpo_cpu_v1/audit-source-2b33abf.tar.xz',
        "cbec0c4f08ce0366556548625972ad59dae502f99fd712b3ab8951d949806489",
    ),
    "5a81dfe532edee93836e8984c007d4bfba4b43ac":(
        ROOT/'benchmarks/streamed_grpo_cpu_v2/audit-source-5a81dfe.tar.xz',
        "daeacb66f13b37697996e3520d9c62ff79c100fc6a965ef2913a1436eae0ef71",
    ),
    "e4155f0d6d7b6781a6d0c3fdcf13a9e3ef1342b7":(
        ROOT/'benchmarks/streamed_grpo_cpu_v3/audit-source-e4155f0.tar.xz',
        "99d2dab3ca4ac206655accfd425641c0483e89183cbb144ee64601598ad30db2",
    ),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    hasher=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_reference_update(path):
    spec=importlib.util.spec_from_file_location('_vare_streamed_reference',path)
    require(spec is not None and spec.loader is not None,"archived reference loader unavailable")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.reference_update


def load_snapshot_store(revision, expected_source_sha256, source_archive=None,
                        source_tree=None, source_files=None):
    """Load VARE from a historical pinned bundle or an anchored run snapshot."""
    require(isinstance(revision,str) and re.fullmatch(r'[0-9a-f]{40}',revision),
            "invalid recorded VARE revision")
    temporary=None
    if revision in PINNED_VARE_SOURCES:
        pinned_archive,archive_sha256=PINNED_VARE_SOURCES[revision]
        source_archive=pinned_archive if source_archive is None else Path(source_archive)
        require(digest(source_archive)==archive_sha256,
                "pinned VARE source archive hash mismatch")
        temporary=tempfile.TemporaryDirectory(prefix='vare-streamed-audit-')
        try:
            with tarfile.open(source_archive,mode='r:xz') as snapshot:
                members=snapshot.getmembers()
                for member in members:
                    name=Path(member.name)
                    require(not name.is_absolute() and '..' not in name.parts
                            and (name.parts==('src',) and member.isdir()
                                 or name.parts[:2]==('src','vare')),
                            "unsafe pinned VARE source archive path")
                    require(member.isdir() or member.isfile(),
                            "unsupported pinned VARE source archive entry")
                snapshot.extractall(temporary.name,filter='data')
        except BaseException:
            temporary.cleanup()
            raise
        source=Path(temporary.name)/'src'/'vare'
        cleanup_source=temporary.cleanup
    else:
        require(source_tree is not None and isinstance(source_files,dict),
                "recorded VARE revision requires an anchored full source snapshot")
        source_tree=Path(source_tree)
        source=source_tree/'src'/'vare'
        expected_names=set()
        for name,expected in source_files.items():
            path=Path(name)
            if name.startswith('src/vare/'):
                require(not path.is_absolute() and '..' not in path.parts
                        and isinstance(expected,str)
                        and re.fullmatch(r'[0-9a-f]{64}',expected),
                        "invalid anchored VARE source entry")
                expected_names.add(name)
                require(digest(source_tree/path)==expected,
                        "anchored VARE source file hash mismatch")
        actual_names={path.relative_to(source_tree).as_posix()
                      for path in source.rglob('*')
                      if path.is_file() and '__pycache__' not in path.parts
                      and path.suffix != '.pyc'}
        require(expected_names and actual_names==expected_names,
                "anchored VARE source snapshot is incomplete")

        def cleanup_source():
            return None

    package_name=f'_vare_snapshot_{uuid4().hex}'
    try:
        require(digest(source/'trajectory_store.py')==expected_source_sha256,
                "recorded revision and archived trajectory store differ")
        package=ModuleType(package_name)
        package.__path__=[str(source)]
        package.__package__=package_name
        sys.modules[package_name]=package
        module=importlib.import_module(f'{package_name}.trajectory_store')
    except BaseException:
        for name in tuple(sys.modules):
            if name==package_name or name.startswith(package_name+'.'):
                del sys.modules[name]
        cleanup_source()
        raise

    def cleanup():
        for name in tuple(sys.modules):
            if name==package_name or name.startswith(package_name+'.'):
                del sys.modules[name]
        cleanup_source()

    return module,cleanup


def tensor_error(actual, expected):
    require(actual.keys()==expected.keys(),"numerical tensor coverage mismatch")
    require(all(actual[name].shape==value.shape and actual[name].dtype==value.dtype for name,value in expected.items()),"numerical tensor shape/dtype mismatch")
    error=max(float((actual[name]-value).abs().max()) for name,value in expected.items())
    require(math.isfinite(error) and error<2e-6,"numerical tensor mismatch")
    return error


def verify_snapshot_source(config, source_files):
    require(config.get('source_sha256') == source_files.get('src/vare/trajectory_store.py'),
            "snapshot runtime source differs from archived source bytes")


def protocol_registry():
    """Return exact frozen protocols and their archived runner locations."""
    return {
        'vare.streamed-grpo-systems.v1': (
            ROOT/'benchmarks/streamed_grpo_cpu_v1/protocol.json',
            'source/scripts/validate_streamed_grpo_cpu.py',
        ),
        'vare.streamed-grpo-systems.v2': (
            ROOT/'benchmarks/streamed_grpo_cpu_v2/protocol.json',
            'source/scripts/validate_streamed_grpo_cpu.py',
        ),
        'vare.streamed-grpo-systems.v3': (
            ROOT/'benchmarks/streamed_grpo_cpu_v3/protocol.json',
            'source/scripts/validate_streamed_grpo_cpu.py',
        ),
        'vare.streamed-grpo-systems.v3.1': (
            ROOT/'benchmarks/streamed_grpo_cpu_v3/protocol-v3.1.json',
            'source/scripts/validate_streamed_grpo_cpu.py',
        ),
        'vare.streamed-grpo-systems.sprint-ii.v1': (
            ROOT/'results/sprint_ii_20261027/streamed_grpo_v32/protocol.json',
            'source/scripts/validate_streamed_grpo_cpu_sprint_ii_v1.py',
        ),
    }


def archived_runner_for_protocol(protocol):
    """Fail closed unless the supplied protocol is byte-for-byte frozen."""
    registry=protocol_registry()
    schema=protocol.get('schema')
    require(schema in registry,"unsupported protocol")
    frozen_protocol,runner_archive=registry[schema]
    require(protocol==json.loads(frozen_protocol.read_text()),"frozen protocol mismatch")
    return runner_archive


def audit(root, model_dir, rvl_source, expected_manifest):
    import torch
    require(not expected_manifest.resolve().is_relative_to(root.resolve()),"external manifest must be outside artifact")
    manifest=json.loads((root/'SHA256.json').read_text())
    require(manifest==json.loads(expected_manifest.read_text()),"external artifact anchor mismatch")
    for name,expected in manifest.items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,"unsafe artifact path")
        require(digest(root/name)==expected,"artifact bytes differ: "+name)
    protocol=json.loads((root/'protocol.json').read_text())
    runner_archive=archived_runner_for_protocol(protocol)
    summary=json.loads((root/'summary.json').read_text())
    for name,expected in summary['source_files'].items():
        require(digest(root/'source'/name)==expected,"source snapshot mismatch")
    require(digest(root/runner_archive)==protocol['runner_sha256'],"runner mismatch")
    reference_path=root/'source/scripts/validate_grpo_e2e_cpu.py'
    reference_update=load_reference_update(reference_path)
    for name,expected in protocol['model_files'].items():
        require(digest(model_dir/name)==expected,"reference model mismatch")
    require(subprocess.check_output(['git','-C',str(rvl_source),'rev-parse','HEAD'],text=True).strip()==protocol['rvl_revision'],"RVL revision mismatch")
    require(subprocess.check_output(['git','-C',str(rvl_source),'status','--porcelain'],text=True).strip()=='M src/rvl_systems/hf_backend.py',"unexpected RVL edits")
    require(digest(rvl_source/'src/rvl_systems/hf_backend.py')==protocol['rvl_patched_backend_sha256'],"RVL patch mismatch")
    sys.path.insert(0,str(rvl_source/'src'))
    from rvl_systems.hf_backend import HFLocalBackend
    from rvl_systems.hf_trainer import HFCausalLMGRPOTrainer,HFTTrainerConfig
    torch.set_num_threads(1)
    backend=HFLocalBackend(str(model_dir),device='cpu',precision='fp32')
    model=backend.model
    for name,parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith('transformer.wpe.') or name.startswith('transformer.ln_f.'))
    config=HFTTrainerConfig(learning_rate=1e-4)
    HFCausalLMGRPOTrainer(model,config=config)  # Reuse the trainer's dropout setup.
    updates=json.loads((root/'updates.json').read_text())
    require(len(updates)==summary['optimizer_calls']==4,"actual update coverage mismatch")
    results=[]
    for update in updates:
        batch=[SimpleNamespace(attempt=SimpleNamespace(**row['attempt']),verification=SimpleNamespace(**row['verification'])) for row in update['batch']]
        for i,row in enumerate(batch):
            row.attempt.created_step=i
        require(len(batch)==protocol['group_size'],"group coverage mismatch")
        before=torch.load(root/(batch[0].attempt.policy_id+'.pt'),weights_only=True)
        candidate=torch.load(root/(update['candidate']+'.pt'),weights_only=True)
        saved=torch.load(root/(update['candidate']+'-gradients.pt'),weights_only=True)
        captured={}
        original=torch.nn.utils.clip_grad_norm_
        def clip(parameters,*args,**kwargs):
            captured['vectors']={n:p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None}
            captured['norm']=float(original(parameters,*args,**kwargs))
            return captured['norm']
        with patch.object(torch.nn.utils,'clip_grad_norm_',clip):
            ref=reference_update(model,before,batch,config)
        norm=update['metrics']['grad_norm']
        require(math.isclose(norm,captured['norm'],rel_tol=2e-6,abs_tol=2e-6),"unclipped norm mismatch")
        coefficient=(config.max_grad_norm/(torch.tensor(norm,dtype=torch.float32)+1e-6)).clamp(max=1)
        reconstructed={n:v/coefficient for n,v in saved['actual_gradients'].items()}
        errors={'preclip_gradient':tensor_error(reconstructed,captured['vectors']),
                'clipped_gradient':tensor_error(saved['actual_gradients'],ref['grads']),
                'parameter':tensor_error(candidate['model'],ref['model'])}
        loss_error=abs(update['metrics']['loss']-ref['loss'])
        require(math.isfinite(loss_error) and loss_error<2e-6,"loss mismatch")
        rng_equal=torch.equal(before['cpu_rng'],candidate['cpu_rng'])
        require(rng_equal,"RNG mismatch")
        require(candidate['optimizer']['param_groups']==ref['optimizer']['param_groups'],"optimizer configuration mismatch")
        require(candidate['optimizer']['state'].keys()==ref['optimizer']['state'].keys(),"optimizer coverage mismatch")
        optimizer_error=max(tensor_error(state,ref['optimizer']['state'][key])
                            for key,state in candidate['optimizer']['state'].items())
        require(optimizer_error<2e-6,"optimizer state mismatch")
        behavior_error=ref['max_behavior_alignment_error']
        require(behavior_error<1e-4,"behavior alignment mismatch")
        errors.update(loss=loss_error,optimizer_state=optimizer_error)
        results.append({'candidate':update['candidate'],'errors':errors,'preclip_norm':norm,
                        'behavior_alignment_error':behavior_error,'rng_equal':rng_equal})
    runtime,cleanup_runtime=load_snapshot_store(
        summary['revision'],summary['source_files']['src/vare/trajectory_store.py'],
        source_tree=root/'source',source_files=summary['source_files'],
    )
    try:
        require(runtime.audit_trajectory_store(root/'snapshot.db')==summary['audit'],"trajectory audit differs")
        store=runtime.TrajectoryStore(root/'snapshot.db',audit_only=True)
        verify_snapshot_source(store.config(),summary['source_files'])
    finally:
        cleanup_runtime()
    rounds=json.loads((root/'rounds.json').read_text())
    require(rounds[1]['result']['decision']['accepted'] is False,"forced rejection missing")
    require(rounds[2]['ack_interrupted'] and rounds[2]['active']['policy_id']==updates[3]['batch'][0]['attempt']['policy_id'],"recovered state not used for next update")
    return {'pass':True,'updates':len(results),'numerics':results,'scope':'Externally anchored bytes and independent full-graph/preclip/Adam/RNG replay. Live checkpoint decoding and logical recovery are separately retained; no task-capability or learner-SIGKILL claim.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--rvl-source',type=Path,required=True)
    parser.add_argument('--expected-manifest',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(audit(args.output,args.model_dir,args.rvl_source,args.expected_manifest),indent=2))
