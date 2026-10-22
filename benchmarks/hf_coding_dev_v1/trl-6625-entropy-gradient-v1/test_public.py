import torch
import torch.nn.functional as F
from trl.trainer.utils import _ChunkedLogProbFunction
def _reference(hidden, weight, labels, temperature):
    logits=(hidden @ weight.T)/temperature
    logp=F.log_softmax(logits, dim=-1)
    entropy=-(logp.exp()*logp).sum(dim=-1)
    chosen=logp.gather(1, labels[:,None]).squeeze(1)
    return chosen, entropy
def test_entropy_gradient():
    for temperature in (1.0, 0.7):
        torch.manual_seed(42)
        hidden=torch.randn(64,32,requires_grad=True)
        weight=torch.randn(257,32,requires_grad=True)
        labels=torch.randint(0,257,(64,))
        _, entropy=_ChunkedLogProbFunction.apply(hidden,weight,labels,temperature,64)
        entropy.sum().backward()
        actual_h, actual_w=hidden.grad.clone(),weight.grad.clone()
        hidden.grad=None; weight.grad=None
        _, ref_entropy=_reference(hidden,weight,labels,temperature)
        ref_entropy.sum().backward()
        torch.testing.assert_close(actual_h,hidden.grad,atol=1e-4,rtol=1e-4)
        torch.testing.assert_close(actual_w,weight.grad,atol=1e-4,rtol=1e-4)
