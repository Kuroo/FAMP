import numpy as np
import math
import torch
from mpi4py import MPI
from torch.optim import Adam


class MpiAdam(Adam):

    @torch.no_grad()
    def step(self, closure=None, comm=None, scale_by_grad_procs=False):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if comm is None:
                    grad = p.grad.data
                else:
                    local_grad = p.grad.view(-1).numpy()
                    global_grad = np.zeros_like(local_grad)
                    comm.Allreduce([local_grad, MPI.FLOAT], [global_grad, MPI.FLOAT], op=MPI.SUM)
                    grad = torch.tensor(global_grad, dtype=torch.get_default_dtype()).view(p.grad.shape)
                    if scale_by_grad_procs:
                        grad /= comm.Get_size()
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size)
        # self.check_sync(comm=comm)
        return loss

    def check_sync(self, comm):
        # Check sync
        for group in self.param_groups:
            for p in group['params']:
                if comm.Get_rank() == 0:
                    global_theta = p.data.view(-1).numpy()
                    comm.Bcast([global_theta, MPI.FLOAT], root=0)
                else:
                    local_theta = p.data.view(-1).numpy()
                    global_theta = np.empty_like(local_theta)
                    comm.Bcast([global_theta, MPI.FLOAT], root=0)
                    assert (global_theta == local_theta).all(), \
                        "Mismatch in params global {}, local {}".format(global_theta, local_theta)


def mpi_adam():
    from torch.optim import Adam
    results = np.zeros((2, 10))

    optimizers = [Adam, MpiAdam]
    names = ["PyTorch Adam", "MPI Adam"]
    for i in range(len(optimizers)):
        np.random.seed(42)
        torch.manual_seed(42)

        a = torch.nn.Parameter(torch.rand(3))
        b = torch.nn.Parameter(torch.rand(2, 5))

        lr = 1e-2
        optimizer = optimizers[i]([a, b], lr=lr)
        for iter in range(10):
            loss = torch.sum(a ** 2 + torch.sum(torch.sin(b)))
            results[i, iter] = loss.item()
            optimizer.zero_grad()
            loss.backward()
            if i == 0:
                optimizer.step()
            else:
                optimizer.step(comm=MPI.COMM_WORLD)
                optimizer.check_sync(comm=MPI.COMM_WORLD)

    print("%s \t %s \t diff" % (names[0], names[1]))
    for i in range(10):
        print("%2.3f \t\t\t %2.3f \t %2.3f" % (results[0][i], results[1][i], np.abs(results[0][i] - results[1][i])))
    assert(np.sum(results[0] - results[1]) == 0), "Difference in Adam implementation"


if __name__ == '__main__':
    mpi_adam()
