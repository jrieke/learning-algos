import numpy as np
import scipy as sp
import scipy.special
import sklearn.metrics
import random
import matplotlib.pyplot as plt
import collections


sigmoid = sp.special.expit
softmax = sp.special.softmax


class Logger:

    def __init__(self):
        self.epoch_values = collections.defaultdict(list)
        self.batch_values = collections.defaultdict(list)

    def log(self, name, value):
        self.batch_values[name].append(value)

    def epoch(self):
        for name in self.batch_values:
            self.epoch_values[name].append(np.mean(self.batch_values[name]))

    def current_average(self):
        return {name: np.mean(self.batch_values[name]) for name in self.batch_values}



def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'."""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def plot_multiple(*arrays_and_labels):
    for arr, lab in arrays_and_labels:
        plt.plot(arr, label=lab)
    plt.legend()
    plt.show()


def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def mean_squared_error(y_pred, y_true):
    return np.mean((y_pred - y_true)**2)


def cross_entropy(y_pred, true_label):
    # TODO: Check if this holds for multiple samples.
    return -np.log(y_pred[true_label])


class NeuralNet:
    """Note that for simplicity, this network uses sigmoid output and the mean squared error loss."""

    def __init__(self, num_hidden=30):
        self.W1 = np.random.randn(784, num_hidden)
        self.W2 = np.random.randn(num_hidden, 10)
        self.b1 = np.random.randn(num_hidden)
        self.b2 = np.random.randn(10)

        # Explicit feedback parameters for feedback alignment, sign symmetry and target propagation.
        self.V2 = np.random.randn(10, num_hidden)
        self.c2 = np.random.randn(num_hidden)

    def forward(self, x):
        # Hidden layer.
        z1 = x @ self.W1 + self.b1
        h1 = sigmoid(z1)

        # Output layer.
        z2 = h1 @ self.W2 + self.b2
        h2 = sigmoid(z2)
        # To use softmax here, do this (when using mini-batches, add axis=1):
        #h2 = softmax(z2)

        return z1, h1, z2, h2

    def backprop_update(self, x, y_true, lr):
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
        self.b1 -= lr * grad_b1
        self.b2 -= lr * grad_b2
        self.W1 -= lr * grad_W1
        self.W2 -= lr * grad_W2

        return h2

    def final_layer_update(self, x, y_true, lr):
        # Run forward pass.
        z1, h1, z2, h2 = self.forward(x)

        # Compute error for final layer (= gradient of cost w.r.t layer input).
        e2 = h2 - y_true  # gradient through cross entropy loss

        # Compute gradients of cost w.r.t. parameters.
        grad_b2 = e2
        grad_W2 = np.outer(h1, e2)

        # Update parameters.
        self.b2 -= lr * grad_b2
        self.W2 -= lr * grad_W2

        return h2

    def feedback_alignment_update(self, x, y_true, lr):
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
        self.b1 -= lr * grad_b1
        self.b2 -= lr * grad_b2
        self.W1 -= lr * grad_W1
        self.W2 -= lr * grad_W2

        return h2

    def sign_symmetry_update(self, x, y_true, lr):
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
        self.b1 -= lr * grad_b1
        self.b2 -= lr * grad_b2
        self.W1 -= lr * grad_W1
        self.W2 -= lr * grad_W2

        return h2

    def weight_mirroring_update(self, x, y_true, lr_forward, lr_backward, weight_decay_backward, only_mirroring=False):
        # "Engaged mode": Forward pass on input image, backward pass to adapt forward weights.
        if not only_mirroring:
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
            self.b1 -= lr_forward * grad_b1
            self.b2 -= lr_forward * grad_b2
            self.W1 -= lr_forward * grad_W1
            self.W2 -= lr_forward * grad_W2


        # "Mirroring mode": Compute activies for random inputs, adapt backward weights.
        x2_noise = np.random.randn(self.W2.shape[0])
        h2_noise = sigmoid(x2_noise @ self.W2 + self.b2)
        self.V2 += lr_backward * np.outer(h2_noise, x2_noise) - lr_backward * weight_decay_backward * self.V2
        # TODO: In the paper, they pass one signal through the network, but subtract the mean at each layer to get 0-mean signals (is this biologically plausible?).

        if not only_mirroring:
            return h2

    def target_prop_update(self, x, y_true, lr_final, lr_forward, lr_backward):
        # Run forward pass.
        z1, h1, z2, h2 = self.forward(x)

        # ---------- Phase 1: Compute targets and change feedforward weights. ----------
        # --------------------- (-> activations approximate targets) -------------------
        # Compute final layer target and backpropagate it.
        h2_target = h2 - lr_final * (h2 - y_true)  # Use the activation given by normal (local!) gradient descent as the last layer target. This is a smoother version than using y_true directly as the target.
        z1_target = h2_target @ self.V2 + self.c2  # Backpropagate the targets.
        h1_target = sigmoid(z1_target)

        # Compute (local) forward losses.
        L1 = mean_squared_error(h1, h1_target)
        L2 = mean_squared_error(h2, h2_target)

        # Compute gradients of forward losses w.r.t. forward parameters.
        dL1_db1 = 2 * (h1 - h1_target) * d_sigmoid(z1)
        dL1_dW1 = 2 * (h1 - h1_target) * np.outer(x, d_sigmoid(z1))  # TODO: Simply by reusing dL1_db1.
        dL2_db2 = 2 * (h2 - h2_target) * d_sigmoid(z2)
        dL2_dW2 = 2 * (h2 - h2_target) * np.outer(h1, d_sigmoid(z2))

        # Update forward parameters.
        self.b1 -= lr_forward * dL1_db1
        self.W1 -= lr_forward * dL1_dW1
        self.b2 -= lr_forward * dL2_db2
        self.W2 -= lr_forward * dL2_dW2


        # ---------- Phase 2: Compute reconstructed activations and change feedback weights. ----------
        # ------------- (-> backward function approximates inverse of forward function) ---------------
        # Compute reconstructed activations (here we only have one feedback connection).
        z1_reconstructed = h2 @ self.V2 + self.c2
        h1_reconstructed = sigmoid(z1_reconstructed)

        # Compute reconstruction loss.
        L_rec1 = mean_squared_error(h1, h1_reconstructed)

        # Compute gradients of reconstruction loss w.r.t. forward parameters.
        dL_rec1_dc2 = 2 * (h1_reconstructed - h1) * d_sigmoid(z1_reconstructed)
        dL_rec1_dV2 = 2 * (h1_reconstructed - h1) * np.outer(h2, d_sigmoid(z1_reconstructed))

        losses = [L1, L2, L_rec1]

        # Update backward parameters.
        self.c2 -= lr_backward * dL_rec1_dc2
        self.V2 -= lr_backward * dL_rec1_dV2

        return h2


    def equilibrium_prop_update(self, x, y_true, lr):
        pass


def evaluate(net, evaluation_data):
    correct = 0
    running_loss = 0
    for i_sample, (x, y_true) in enumerate(evaluation_data):
        x = x.flatten()
        y_true = y_true.flatten()
        _, _, _, y_pred = net.forward(x)

        true_label = y_true[0]  # target values in validation_data are labels (in training_data they are one-hot vectors)
        running_loss += cross_entropy(y_pred, true_label)
        if true_label == y_pred.argmax():
            correct += 1

    avg_loss = running_loss / len(evaluation_data)
    avg_acc = correct / len(evaluation_data) * 100

    print('Evaluating:\tLoss: {:.4f}\tAccuracy: {:.1f}%\n'.format(avg_loss, avg_acc))


def train_mirroring_epoch(net, num_samples, lr_backward, weight_decay_backward):
    angle_before = np.rad2deg(angle_between(net.V2.flatten(), net.W2.T.flatten()))
    mean_before = net.V2.mean()
    for i_sample in range(num_samples):
        net.weight_mirroring_update(None, None, lr_forward=0, lr_backward=lr_backward, weight_decay_backward=weight_decay_backward, only_mirroring=True)
    angle_after = np.rad2deg(angle_between(net.V2.flatten(), net.W2.T.flatten()))
    mean_after = net.V2.mean()
    print(f'Ran mirroring for one epoch, reduced angle between V2 and W2.T from {angle_before}° to {angle_after}°, mean of V2 changed from {mean_before} to {mean_after}')



def train_epoch(net, training_data, lr, shuffle=True):
    if shuffle:
        random.shuffle(training_data)
    correct = 0
    running_loss = 0
    for i_sample, (x, y_true) in enumerate(training_data):
        x = x.flatten()
        y_true = y_true.flatten()
        y_pred = net.target_prop_update(x, y_true, 0.5, 0.3, 0.001)#lr)
        #y_pred = net.feedback_alignment_update(x, y_true, 0.2)
        #y_pred = net.weight_mirroring_update(x, y_true, lr_forward=0.1, lr_backward=0.005, weight_decay_backward=0.2)

        true_label = y_true.argmax()  # target values in training_data they are one-hot vectors (in validation_data they are labels)
        running_loss += cross_entropy(y_pred, true_label)
        if true_label == y_pred.argmax():
            correct += 1

        if i_sample % 5000 == 1:
            print('{} / {} samples - loss: {:.6f} - accuracy: {:.1f} %'.format(i_sample, len(training_data), running_loss / i_sample, correct / i_sample * 100))

    print('Average:\tLoss: {:.6f}\tAccuracy: {:.1f}%'.format(running_loss / len(training_data), correct / len(training_data) * 100))
    print('Angle between V2 and W2.T', np.rad2deg(angle_between(net.V2.flatten(), net.W2.T.flatten())))
    print('Mean of V2 ', net.V2.mean())
