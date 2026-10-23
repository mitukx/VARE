import torch.distributed
from accelerate import PartialState
from accelerate.utils import DistributedType
def test_multi_cpu_barrier():
    state=PartialState()
    state.__dict__["distributed_type"]=DistributedType.MULTI_CPU
    state.__dict__["local_process_index"]=0
    calls=[]
    original=torch.distributed.barrier
    torch.distributed.barrier=lambda *args, **kwargs: calls.append((args,kwargs))
    try:
        state.wait_for_everyone()
    finally:
        torch.distributed.barrier=original
    assert calls == [((),{})], calls
