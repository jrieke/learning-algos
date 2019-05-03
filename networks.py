import numpy as np
import scipy as sp
import scipy.special
import utils


sigmoid = sp.special.expit


def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def mean_squared_error(y_pred, y_true):
    return np.mean((y_pred - y_true)**2)


class BaseNet:
    """A basic feedforward net for MNIST with one hidden layer."""

    def __init__(self, num_hidden=30):
        self.W1 = np.random.randn(784, num_hidden)
        self.W2 = np.random.randn(num_hidden, 10)
        self.b1 = np.random.randn(num_hidden)
        self.b2 = np.random.randn(10)


    def forward(self, x):
        # Hidden layer.
        z1 = x @ self.W1 + self.b1
        h1 = sigmoid(z1)

        # Output layer.
        z2 = h1 @ self.W2 + self.b2
        h2 = sigmoid(z2)

        return z1, h1, z2, h2


class FinalLayerUpdateNet(BaseNet):
    """A net which only updates the final but not the hidden layer (i.e. it doesn't require backprop!)."""

    def update(self, x, y_true, params, averager=None):
        # Run forward pass.
        z1, h1, z2, h2 = self.forward(x)

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
        z1, h1, z2, h2 = self.forward(x)

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
        z1, h1, z2, h2 = self.forward(x)

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
        z1, h1, z2, h2 = self.forward(x)

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
        z1, h1, z2, h2 = self.forward(x)

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
        z1, h1, z2, h2 = self.forward(x)

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
