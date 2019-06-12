import numpy as np
import scipy as sp
import scipy.special
import utils


sigmoid = sp.special.expit


def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def hard_sigmoid(x):
    return np.clip(x, 0, 1)


def d_hard_sigmoid(x):
    return ((x >= 0) * (x <= 1)).astype(x.dtype)


def mean_squared_error(y_pred, y_true):
    return np.mean((y_pred - y_true)**2)


class BaseNet:
    """A basic feedforward net for MNIST with one hidden layer."""

    def __init__(self, num_hidden=30):
        self.W1 = np.random.randn(784, num_hidden)
        self.W2 = np.random.randn(num_hidden, 10)
        self.b1 = np.random.randn(num_hidden)
        self.b2 = np.random.randn(10)

    def forward(self, x, params, return_activations=False):
        # Hidden layer.
        z1 = x @ self.W1 + self.b1
        h1 = sigmoid(z1)

        # Output layer.
        z2 = h1 @ self.W2 + self.b2
        h2 = sigmoid(z2)

        if return_activations:
            return z1, h1, z2, h2
        else:
            return h2


class FinalLayerUpdateNet(BaseNet):
    """A net which only updates the final but not the hidden layer (i.e. it doesn't require backprop!)."""

    def update(self, x, y_true, params, averager=None):
        # Run forward pass.
        z1, h1, z2, h2 = self.forward(x, params, return_activations=True)

        # Compute error for final layer (= gradient of cost w.r.t layer input).
        e2 = h2 - y_true  # gradient through cross entropy loss

        # Compute gradients of cost w.r.t. parameters.
        grad_b2 = e2
        grad_W2 = np.outer(h1, e2)

        # Update parameters.
        self.b2 -= params['lr'] * grad_b2
        self.W2 -= params['lr'] * grad_W2

        return h2


class BackpropagationNet(BaseNet):
    """A net with standard backpropagation updates."""

    def update(self, x, y_true, params, averager=None):
        # Run forward pass.
        z1, h1, z2, h2 = self.forward(x, params, return_activations=True)

        # Compute errors for each layer (= gradient of cost w.r.t layer input).
        e2 = h2 - y_true  # gradient through cross entropy loss
        e1 = d_sigmoid(z1) * (e2 @ self.W2.T)  # gradient backpropagation

        # Compute gradients of cost w.r.t. parameters.
        grad_b1 = e1
        grad_b2 = e2
        grad_W1 = np.outer(x, e1)  # np.outer creates a matrix from two vectors
        grad_W2 = np.outer(h1, e2)

        # Update parameters.
        self.b1 -= params['lr'] * grad_b1
        self.b2 -= params['lr'] * grad_b2
        self.W1 -= params['lr'] * grad_W1
        self.W2 -= params['lr'] * grad_W2

        return h2


class FeedbackAlignmentNet(BaseNet):
    """A net with feedback alignment updates (Lillicrap et al. 2016)."""

    def __init__(self, num_hidden=30):
        super().__init__(num_hidden)
        self.V2 = np.random.randn(10, num_hidden)  # explicit feedback connections
        self.c2 = np.random.randn(num_hidden)

    def update(self, x, y_true, params, averager=None):
        # Run forward pass.
        z1, h1, z2, h2 = self.forward(x, params, return_activations=True)

        # Compute errors for each layer (= gradient of cost w.r.t layer input).
        e2 = h2 - y_true  # gradient through cross entropy loss
        e1 = d_sigmoid(z1) * (e2 @ self.V2)  # gradient backpropagation

        # Compute gradients of cost w.r.t. parameters.
        grad_b1 = e1
        grad_b2 = e2
        grad_W1 = np.outer(x, e1)  # np.outer creates a matrix from two vectors
        grad_W2 = np.outer(h1, e2)

        # Update parameters.
        self.b1 -= params['lr'] * grad_b1
        self.b2 -= params['lr'] * grad_b2
        self.W1 -= params['lr'] * grad_W1
        self.W2 -= params['lr'] * grad_W2

        averager.add('backward_angle', np.rad2deg(utils.angle_between(self.V2.flatten(), self.W2.T.flatten())))

        return h2


class SignSymmetryNet(BaseNet):
    """A net with sign symmetry updates (Liao et al. 2016)."""

    def __init__(self, num_hidden=30):
        super().__init__(num_hidden)
        self.V2 = np.random.randn(10, num_hidden)  # explicit feedback connections
        self.c2 = np.random.randn(num_hidden)

    def update(self, x, y_true, params, averager=None):
        # Run forward pass.
        z1, h1, z2, h2 = self.forward(x, params, return_activations=True)

        # Compute errors for each layer (= gradient of cost w.r.t layer input).
        e2 = h2 - y_true  # gradient through cross entropy loss
        e1 = d_sigmoid(z1) * (e2 @ (np.abs(self.V2) * np.sign(self.W2.T)))  # gradient backpropagation

        # Using these errors, compute gradients of cost w.r.t. parameters.
        grad_b1 = e1
        grad_b2 = e2
        grad_W1 = np.outer(x, e1)  # np.outer creates a matrix from two vectors
        grad_W2 = np.outer(h1, e2)

        # Update parameters.
        self.b1 -= params['lr'] * grad_b1
        self.b2 -= params['lr'] * grad_b2
        self.W1 -= params['lr'] * grad_W1
        self.W2 -= params['lr'] * grad_W2

        averager.add('backward_angle', np.rad2deg(utils.angle_between((np.abs(self.V2) * np.sign(self.W2.T)).flatten(), self.W2.T.flatten())))

        return h2


class WeightMirroringNet(BaseNet):
    """A net with weight mirroring updates (Wilson et al. 2019)."""

    def __init__(self, num_hidden=30):
        super().__init__(num_hidden)
        self.V2 = np.random.randn(10, num_hidden)  # explicit feedback connections
        self.c2 = np.random.randn(num_hidden)

    def update(self, x, y_true, params, averager=None):
        # "Engaged mode": Forward pass on input image, backward pass to adapt forward weights.
        # Run forward pass.
        z1, h1, z2, h2 = self.forward(x, params, return_activations=True)

        # Compute errors for each layer (= gradient of cost w.r.t layer input).
        e2 = h2 - y_true  # gradient through cross entropy loss
        e1 = d_sigmoid(z1) * (e2 @ self.V2)  # gradient backpropagation

        # Compute gradients of cost w.r.t. parameters.
        grad_b1 = e1
        grad_b2 = e2
        grad_W1 = np.outer(x, e1)  # np.outer creates a matrix from two vectors
        grad_W2 = np.outer(h1, e2)

        # Update parameters.
        self.b1 -= params['lr_forward'] * grad_b1
        self.b2 -= params['lr_forward'] * grad_b2
        self.W1 -= params['lr_forward'] * grad_W1
        self.W2 -= params['lr_forward'] * grad_W2

        # "Mirroring mode": Compute activies for random inputs, adapt backward weights.
        self.update_weight_mirror(params)

        averager.add('backward_angle', np.rad2deg(utils.angle_between(self.V2.flatten(), self.W2.T.flatten())))
        averager.add('backward_mean', np.mean(self.V2.flatten()))

        return h2

    def update_weight_mirror(self, params):
        x2_noise = np.random.randn(self.W2.shape[0])
        h2_noise = sigmoid(x2_noise @ self.W2 + self.b2)
        self.V2 += params['lr_backward'] * np.outer(h2_noise, x2_noise) - params['lr_backward'] * params[
            'weight_decay_backward'] * self.V2
        # TODO: In the paper, they pass one signal through the network, but subtract the mean at each layer to get 0-mean signals (is this biologically plausible?).


class TargetPropagationNet(BaseNet):
    """A net with target propagation updates (Lee et al. 2015). Not that this does not implement DTP but "vanilla" TP."""

    def __init__(self, num_hidden=30):
        super().__init__(num_hidden)
        self.V2 = np.random.randn(10, num_hidden)  # explicit feedback connections
        self.c2 = np.random.randn(num_hidden)

    def update(self, x, y_true, params, averager=None):
        # Run forward pass.
        z1, h1, z2, h2 = self.forward(x, params, return_activations=True)

        # ---------- Phase 1: Compute targets and change feedforward weights. ----------
        # --------------------- (-> activations approximate targets) -------------------
        # Compute final layer target and backpropagate it.
        h2_target = h2 - params['lr_final'] * (h2 - y_true)  # Use the activation given by normal (local!) gradient descent as the last layer target. This is a smoother version than using y_true directly as the target.
        z1_target = h2_target @ self.V2 + self.c2  # Backpropagate the targets.
        h1_target = sigmoid(z1_target)

        # Compute (local) forward losses.
        L1 = mean_squared_error(h1, h1_target)
        L2 = mean_squared_error(h2, h2_target)
        averager.add('L1', L1)
        averager.add('L2', L2)

        # Compute gradients of forward losses w.r.t. forward parameters.
        dL1_db1 = 2 * (h1 - h1_target) * d_sigmoid(z1)
        dL1_dW1 = 2 * (h1 - h1_target) * np.outer(x, d_sigmoid(z1))  # TODO: Simply by reusing dL1_db1.
        dL2_db2 = 2 * (h2 - h2_target) * d_sigmoid(z2)
        dL2_dW2 = 2 * (h2 - h2_target) * np.outer(h1, d_sigmoid(z2))

        # Update forward parameters.
        self.b1 -= params['lr_forward'] * dL1_db1
        self.W1 -= params['lr_forward'] * dL1_dW1
        self.b2 -= params['lr_forward'] * dL2_db2
        self.W2 -= params['lr_forward'] * dL2_dW2


        # ---------- Phase 2: Compute reconstructed activations and change feedback weights. ----------
        # ------------- (-> backward function approximates inverse of forward function) ---------------
        # Compute reconstructed activations (here we only have one feedback connection).
        z1_reconstructed = h2 @ self.V2 + self.c2
        h1_reconstructed = sigmoid(z1_reconstructed)

        # Compute reconstruction loss.
        L_rec1 = mean_squared_error(h1, h1_reconstructed)
        averager.add('L_rec1', L_rec1)

        # Compute gradients of reconstruction loss w.r.t. forward parameters.
        dL_rec1_dc2 = 2 * (h1_reconstructed - h1) * d_sigmoid(z1_reconstructed)
        dL_rec1_dV2 = 2 * (h1_reconstructed - h1) * np.outer(h2, d_sigmoid(z1_reconstructed))

        # Update backward parameters.
        self.c2 -= params['lr_backward'] * dL_rec1_dc2
        self.V2 -= params['lr_backward'] * dL_rec1_dV2

        averager.add('backward_angle', np.rad2deg(utils.angle_between(self.V2.flatten(), self.W2.T.flatten())))
        averager.add('backward_mean', np.mean(self.V2.flatten()))

        return h2


class EquilibriumPropagationNet(BaseNet):
    """
    A continuous net with equilibrium propagation updates (Scellier & Bengio 2017).

    Note that there is an explicit recurrent connection between output and hidden layer with weight self.W2.T.

    # TODO
    Differences to paper:
    - 30 instead of 500 hidden neurons
    - weight initialization via np.random.randn instead of Glorot
    - no persistent particles
    - no mini-batching with batch size 20, implement this via jax
    - sigmoid instead of hard-sigmoid (+clipping of u). With hard sigmoid, u1 and u2 also settle but they become binary. CHECK AGAIN IF THIS ALSO TRAINS PROPERLY.
    """

    def settle(self, u1, u2, x, y_true, steps, step_size, beta):
        for step in range(steps):
            rho_u1 = sigmoid(u1)
            rho_u2 = sigmoid(u2)

            E = 0.5 * (np.sum(u1**2) + np.sum(u2**2)) - 0.5 * (x @ self.W1 @ rho_u1 + rho_u1 @ self.W2 @ rho_u2 + rho_u2 @ self.W2.T @ rho_u1) - (np.sum(self.b1 * rho_u1) + np.sum(self.b2 * rho_u2))
            C = 0.5 * np.sum((rho_u2 - y_true) ** 2)  # TODO: Maybe omit this calculation if beta == 0. Check if this increases performance.
            F = E + beta * C

            u1_old, u2_old = u1.copy(), u2.copy()

            dE_du1 = -d_sigmoid(u1) * (x @ self.W1 + rho_u2 @ self.W2.T + self.b1) + u1
            u1 -= step_size * dE_du1
            #u1 = np.clip(u1, 0, 1)  # critical in combination with hard sigmoid, see equation 43 in Scellier & Bengio 2017

            dE_du2 = -d_sigmoid(u2) * (rho_u1 @ self.W2 + self.b2) + u2
            dC_du2 = rho_u2 - y_true  # TODO: Maybe omit this calculation if beta == 0. Check if this increases performance.
            u2 -= step_size * (dE_du2 + beta * dC_du2)
            #u2 = np.clip(u2, 0, 1)

            change_u1 = np.sum(np.abs(u1 - u1_old))
            change_u2 = np.sum(np.abs(u2 - u2_old))

        return u1, u2, E, C, F

    # Interestingly, with the forward method of a standard MLP, this also achieves pretty good results, even a bit better than for the continuous forward method ;)
    # TODO: Refactor this and base method.
    def forward(self, x, params):
        u1 = np.random.randn(len(self.b1))
        u2 = np.random.randn(len(self.b2))

        u1, u2, E_0, C_0, F_0 = self.settle(u1, u2, x, 0, params['steps_free'], params['step_size'], beta=0)

        return sigmoid(u2)

    def update(self, x, y_true, params, averager=None):

        u1 = np.random.randn(len(self.b1))
        u2 = np.random.randn(len(self.b2))


        # Step 1, free phase: Clamp x, relax to fixed point, collect prediction and derivative dF_dW.
        u1, u2, E_0, C_0, F_0 = self.settle(u1, u2, x, y_true, params['steps_free'], params['step_size'], beta=0)

        rho_u1 = sigmoid(u1)
        rho_u2 = sigmoid(u2)

        y_pred = rho_u2.copy()  # return this later

        dF_0_dW1 = np.outer(x, rho_u1)
        dF_0_dW2 = np.outer(rho_u1, rho_u2)
        dF_0_db1 = rho_u1.copy()
        dF_0_db2 = rho_u2.copy()

        averager.add('E_0', E_0)
        averager.add('C_0', C_0)

        # Results: This settles to a fixed point (change_u1 and change_u2 decreases to 0) at a minimum of E (E decreases during the first few iterations, converges).
        #          Didn't test hard sigmoid with the new implementation yet.


        # Step 2, weakly clamped phase: Clamp y_true, relax to fixed point s_beta, collect derivative dF_beta_dW.
        beta = np.random.choice([-1, 1]) * params['beta']  # choose sign of beta at random for the 2nd phase, this helps learning according to the paper
        u1, u2, E_beta, C_beta, F_beta = self.settle(u1, u2, x, y_true, params['steps_clamped'], params['step_size'], beta=beta)

        rho_u1 = sigmoid(u1)
        rho_u2 = sigmoid(u2)

        dF_beta_dW1 = np.outer(x, rho_u1)
        dF_beta_dW2 = np.outer(rho_u1, rho_u2)
        # TODO: In principle, this would need another gradient for the recurrent connection, which would be the same as dF_dW2.T though.
        dF_beta_db1 = rho_u1.copy()
        dF_beta_db2 = rho_u2.copy()

        averager.add('E_beta', E_beta)
        averager.add('C_beta', C_beta)

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
