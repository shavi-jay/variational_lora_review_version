import torch
import os
from packaging import version


def traceable(cls):
    """Decorator over customer functions
    There is an issue for tracing customer python torch Function, using this decorator to work around it.
    e.g.
    @traceable
    class MyOp(torch.autograd.Function):
    xxx
    """

    class _Function(object):
        @staticmethod
        def apply(*args):
            jit_trace = os.getenv("JIT_TRACE", "False").lower() == "true"
            if jit_trace:
                return cls.forward(_Function, *args)
            else:
                return cls.apply(*args)

        @staticmethod
        def save_for_backward(*args):
            pass

    _Function.__name__ = cls.__name__
    _Function.__doc__ = cls.__doc__
    return _Function


class DropoutContext(object):
    def __init__(self):
        self.dropout = 0
        self.mask = None
        self.scale = 1
        self.reuse_mask = True


def get_mask(input, local_context):
    if not isinstance(local_context, DropoutContext):
        dropout = local_context
        mask = None
    else:
        dropout = local_context.dropout
        dropout *= local_context.scale
        mask = local_context.mask if local_context.reuse_mask else None

    if dropout > 0 and mask is None:
        if version.Version(torch.__version__) >= version.Version("1.2.0a"):
            mask = (1 - torch.empty_like(input).bernoulli_(1 - dropout)).bool()
        else:
            mask = (1 - torch.empty_like(input).bernoulli_(1 - dropout)).byte()

    if isinstance(local_context, DropoutContext):
        if local_context.mask is None:
            local_context.mask = mask

    return mask, dropout


@traceable
class XDropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, local_ctx):
        mask, dropout = get_mask(input, local_ctx)
        ctx.scale = 1.0 / (1 - dropout)
        if dropout > 0:
            ctx.save_for_backward(mask)
            return input.masked_fill(mask, 0) * ctx.scale
        else:
            return input

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.scale > 1:
            (mask,) = ctx.saved_tensors
            return grad_output.masked_fill(mask, 0) * ctx.scale, None
        else:
            return grad_output, None


class StableDropout(torch.nn.Module):
    """Optimized dropout module for stabilizing the training

    Args:

      drop_prob (float): the dropout probabilities

    """

    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob
        self.count = 0
        self.context_stack = None

    def forward(self, x):
        """Call the module

        Args:

          x (:obj:`torch.tensor`): The input tensor to apply dropout


        """
        if self.training and self.drop_prob > 0:
            return XDropout.apply(x, self.get_context())
        return x

    def clear_context(self):
        self.count = 0
        self.context_stack = None

    def init_context(self, reuse_mask=True, scale=1):
        if self.context_stack is None:
            self.context_stack = []
        self.count = 0
        for c in self.context_stack:
            c.reuse_mask = reuse_mask
            c.scale = scale

    def get_context(self):
        if self.context_stack is not None:
            if self.count >= len(self.context_stack):
                self.context_stack.append(DropoutContext())
            ctx = self.context_stack[self.count]
            ctx.dropout = self.drop_prob
            self.count += 1
            return ctx
        else:
            return self.drop_prob


def MaskedLayerNorm(layerNorm, input, mask=None):
    """Masked LayerNorm which will apply mask over the output of LayerNorm to avoid inaccurate updatings to the LayerNorm module.

    Args:
      layernorm (:obj:`~DeBERTa.deberta.LayerNorm`): LayerNorm module or function
      input (:obj:`torch.tensor`): The input tensor
      mask (:obj:`torch.IntTensor`): The mask to applied on the output of LayerNorm where `0` indicate the output of that element will be ignored, i.e. set to `0`

    Example::

      # Create a tensor b x n x d
      x = torch.randn([1,10,100])
      m = torch.tensor([[1,1,1,0,0,0,0,0,0,0]], dtype=torch.int)
      LayerNorm = DeBERTa.deberta.LayerNorm(100)
      y = MaskedLayerNorm(LayerNorm, x, m)

    """
    output = layerNorm(input).to(input)
    if mask is None:
        return output
    if mask.dim() != input.dim():
        if mask.dim() == 4:
            mask = mask.squeeze(1).squeeze(1)
        mask = mask.unsqueeze(2)
    mask = mask.to(output.dtype)
    return output * mask


def swish(x):
    return x * torch.sigmoid(x)


def linear_act(x):
    return x


ACT2FN = {
    "gelu": torch.nn.functional.gelu,
    "relu": torch.nn.functional.relu,
    "swish": swish,
    "tanh": torch.tanh,
    "linear": linear_act,
    "sigmoid": torch.sigmoid,
}
