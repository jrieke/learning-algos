import numpy as np
import random
import scipy as sp
import scipy.special
import utils
import torch
import torch.nn as nn


class EquilibriumPropagationNet(nn.Module):
    """
    A continuous net with equilibrium propagation updates (Scellier & Bengio 2017).

    Note that there is an explicit recurrent connection between output and hidden layer with weight self.W2.T.

    # TODO
    Differences to paper:
    - 30 instead of 500 hidden neurons
    - weight initialization via random normal instead of Glorot
    - no persistent particles
    - no mini-batching with batch size 20, implement this via jax
    - sigmoid instead of hard-sigmoid (+clipping of u). With hard sigmoid, u1 and u2 also settle but they become binary. CHECK AGAIN IF THIS ALSO TRAINS PROPERLY.
    """

    def __init__(self, num_hidden=30):
        self.W1 = nn.Parameter(torch.Tensor(784, num_hidden))
        self.W2 = nn.Parameter(torch.Tensor(num_hidden, 10))
        self.b1 = nn.Parameter(torch.Tensor(num_hidden))
        self.b2 = nn.Parameter(torch.Tensor(10))

        # TODO: Change to Xavier or Kaiming initialization.
        nn.init.normal_(self.W1)
        nn.init.normal_(self.W2)
        nn.init.normal_(self.b1)
        nn.init.normal_(self.b2)

    def settle(self, u1, u2, x, y_true, steps, step_size, beta):

        # TODO: Store the requires_grad states here and restore them later, so we can set weights to non-trainable.
        for p in self.parameters():
            p.requires_grad = False
        u1.requires_grad = True
        u2.requires_grad = True

        for step in range(steps):
            rho_u1 = nn.Sigmoid(u1)
            rho_u2 = nn.Sigmoid(u2)

            # TODO: Check that matmul stuff with @ operator works.
            E = 0.5 * (torch.sum(u1**2) + torch.sum(u2**2)) - 0.5 * (x @ self.W1 @ rho_u1 + rho_u1 @ self.W2 @ rho_u2 + rho_u2 @ self.W2.T @ rho_u1) - (torch.sum(self.b1 * rho_u1) + torch.sum(self.b2 * rho_u2))
            C = 0.5 * torch.sum((rho_u2 - y_true) ** 2)  # TODO: Maybe omit this calculation if beta == 0. Check if this increases performance.
            F = E + beta * C

            u1_old, u2_old = torch.Tensor(u1), torch.Tensor(u2)

            # dE_du1 = -d_sigmoid(u1) * (x @ self.W1 + rho_u2 @ self.W2.T + self.b1) + u1
            # u1 -= step_size * dE_du1
            #u1 = np.clip(u1, 0, 1)  # critical in combination with hard sigmoid, see equation 43 in Scellier & Bengio 2017

            # dE_du2 = -d_sigmoid(u2) * (rho_u1 @ self.W2 + self.b2) + u2
            # dC_du2 = rho_u2 - y_true  # TODO: Maybe omit this calculation if beta == 0. Check if this increases performance.
            # u2 -= step_size * (dE_du2 + beta * dC_du2)
            #u2 = np.clip(u2, 0, 1)

            F.backward()
            u1 -= step_size * u1.grad
            u2 -= step_size * u2.grad
            u1.grad.zero_()
            u2.grad.zero_()

            change_u1 = torch.sum(torch.abs(u1 - u1_old))
            change_u2 = torch.sum(torch.abs(u2 - u2_old))

        u1.requires_grad = False
        u2.requires_grad = False
        for p in self.parameters():
            p.requires_grad = True

        return u1, u2, E, C, F

    # Interestingly, with the forward method of a standard MLP, this also achieves pretty good results, even a bit better than for the continuous forward method ;)
    # TODO: Refactor this and base method.
    def forward(self, x, params):
        # TODO: Do this in training loop.
        x = torch.from_numpy(x)

        u1 = torch.randn(len(self.b1))
        u2 = torch.randn(len(self.b2))

        u1, u2, E_0, C_0, F_0 = self.settle(u1, u2, x, 0, params['steps_free'], params['step_size'], beta=0)

        rho_u2 = nn.Sigmoid(u2)
        y_pred = rho_u2.clone().detach().numpy()
        return y_pred

    def update(self, x, y_true, params, averager=None):
        u1 = torch.randn(len(self.b1))
        u2 = torch.randn(len(self.b2))

        # TODO: Do this in training loop.
        x = torch.from_numpy(x)
        y_true = torch.from_numpy(y_true)


        # Step 1, free phase: Clamp x, relax to fixed point, collect prediction and derivative dF_dW.
        u1, u2, E_0, C_0, F_0 = self.settle(u1, u2, x, y_true, params['steps_free'], params['step_size'], beta=0)

        rho_u2 = nn.Sigmoid(u2)
        y_pred = rho_u2.clone().detach().numpy()  # return this later

        F_0.backward()
        # TODO: Store gradients w.r.t. parameters, either in grad fields or outside.


        averager.add('E_0', E_0.item())
        averager.add('C_0', C_0.item())

        # Results: This settles to a fixed point (change_u1 and change_u2 decreases to 0) at a minimum of E (E decreases during the first few iterations, converges).
        #          Didn't test hard sigmoid with the new implementation yet.


        # Step 2, weakly clamped phase: Clamp y_true, relax to fixed point s_beta, collect derivative dF_beta_dW.
        beta = random.choice([-1, 1]) * params['beta']  # choose sign of beta at random for the 2nd phase, this helps learning according to the paper
        u1, u2, E_beta, C_beta, F_beta = self.settle(u1, u2, x, y_true, params['steps_clamped'], params['step_size'], beta=beta)

        F_beta.backward()
        # TODO: Store gradients w.r.t. parameters, either in grad fields or outside.

        averager.add('E_beta', E_beta.item())
        averager.add('C_beta', C_beta.item())

        # Results: This settles to a fixed point (change_u2 is high from the beginning, change_u2 becomes high at 2nd iteration, both change_u1 and change_u2 decrease, but do not completely converge within the 4 iterations)
        #          at a new minimum of the energy (E converges) which is a bit higher compared to step 1.
        #          The cost at this new fixed point is lower than at the original fixed point (C decreases and converges but doesn't go to 0, e.g. from 1.2 to 0.8).


        # Step 3: Update network weights according to the update rule.
        self.W1 += params['lr1'] / beta * (dF_beta_dW1 - dF_0_dW1)  # TODO: Maybe do this *0,5 to make it more similar to theory.
        self.W2 += params['lr2'] / beta * (dF_beta_dW2 - dF_0_dW2)  # TODO: This could stay like it is though to incorporate the recurrent connection.

        # TODO: Should be correct according to theory. Check that this in fact improves implementation. Seed everything before.
        self.b1 += params['lr1'] / beta * (dF_beta_db1 - dF_0_db1)
        self.b2 += params['lr2'] / beta * (dF_beta_db2 - dF_0_db2)

        return y_pred
