import torch
from torch.autograd import Function
import scipy.signal
import numpy as np
from typing import Any


class DiscountedSumBackward(Function):
    @staticmethod
    def forward(ctx, values, discount) -> Any:
        # detach so we can cast to NumPy
        values = values.detach()
        discount = torch.tensor([discount])
        ctx.save_for_backward(values, discount)
        discount = discount.item()
        result = torch.as_tensor(scipy.signal.lfilter([1], [1, -discount], np.flip(values.numpy(), axis=0),
                                                      axis=0), dtype=values.dtype).flip(dims=(0,))
        return result

    @staticmethod
    def backward(ctx, grad_output):
        values, discount = ctx.saved_tensors
        grad_values = grad_discount = None
        if ctx.needs_input_grad[0]:
            grad_values = DiscountedSumForward.apply(grad_output, discount)
        if ctx.needs_input_grad[1]:
            raise NotImplementedError
        return grad_values, grad_discount

    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        raise NotImplementedError("JVP is not implemented for this function")


class DiscountedSumForward(Function):
    @staticmethod
    def forward(ctx, values, discount):
        # detach so we can cast to NumPy
        values = values.detach()
        discount = torch.tensor([discount])
        ctx.save_for_backward(values, discount)
        discount = discount.item()
        result = torch.as_tensor(scipy.signal.lfilter([1], [1, -discount], values.numpy(), axis=0), dtype=values.dtype)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        values, discount = ctx.saved_tensors
        grad_values = grad_discount = None
        if ctx.needs_input_grad[0]:
            grad_values = DiscountedSumBackward.apply(grad_output, discount)
        if ctx.needs_input_grad[1]:
            raise NotImplementedError
        return grad_values, grad_discount

    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        raise NotImplementedError("JVP is not implemented for this function")


if __name__ == '__main__':
    funcs = [DiscountedSumBackward.apply, DiscountedSumForward.apply]
    for func in funcs:
        torch.manual_seed(42)
        rewards = torch.ones((6, 1), requires_grad=True, dtype=torch.double)
        dsc = torch.tensor([0.9], dtype=torch.double)
        a = func(rewards, dsc)
        torch.autograd.gradcheck(func, (rewards, dsc))
        torch.autograd.gradgradcheck(func=func, inputs=(rewards, dsc), grad_outputs=torch.ones((6, 1), requires_grad=True, dtype=torch.double) * 2)
