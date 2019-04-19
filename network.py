import numpy as np
import scipy as sp
import scipy.special
import sklearn.metrics
import random


sigmoid = sp.special.expit
softmax = sp.special.softmax


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

    # def backprop_update(self, batch, lr):
    #     # Average gradients over the mini batch.
    #     grads_b1, grads_b2, grads_W1, grads_W2 = 0, 0, 0, 0
    #     outputs = []
    #     for x, y_true in batch:
    #         x = x.flatten()
    #         y_true = y_true.flatten()
    #
    #         # Run forward pass.
    #         z1, a1, z2, a2 = self.forward(x)
    #         outputs.append(a2)
    #
    #         # Compute errors for each layer (= gradient of cost w.r.t layer input).
    #         e2 = a2 - y_true  # gradient through cross entropy and softmax
    #         e1 = d_sigmoid(z2) * e2 @ self.W2.T  # gradient backpropagation
    #
    #         # Compute gradients of cost w.r.t. parameters.
    #         grads_b1 += e1
    #         grads_b2 += e2
    #         grads_W1 += np.outer(x, e1)  # np.outer creates a matrix from two vectors
    #         grads_W2 += np.outer(a1, e2)
    #
    #     # Update parameters.
    #     self.b1 -= lr * grads_b1 / len(batch)
    #     self.b2 -= lr * grads_b2 / len(batch)
    #     self.W1 -= lr * grads_W1 / len(batch)
    #     self.W2 -= lr * grads_W2 / len(batch)
    #
    #     return np.array(outputs)

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

    def weight_mirroring_update(self, x, y_true, lr):
        pass

    def target_prop_update(self, x, y_true, lr_final, lr_forward, lr_backward):
        # Run forward pass.
        # TODO: Rename a2, a2 to h1, h2 everywhere.
        z1, h1, z2, h2 = self.forward(x)

        # Compute targets for each layer.
        h2_ = h2 - lr_final * (h2 - y_true)  # TODO: Why not use y_true here directly?
        z1_ = h2_ @ self.V2 + self.c2  # TODO: Use h2 or h2_ here?
        h1_ = sigmoid(z1_)

        # Compute layerwise loss.
        L2 = mean_squared_error(h2, h2_)
        L1 = mean_squared_error(h1, h1_)
        #print(L2, L1)

        # Compute local gradients for forward parameters.
        dL1_db1 = 2 * np.linalg.norm(h1 - h1_) * d_sigmoid(z1)
        dL1_dW1 = 2 * np.linalg.norm(h1 - h1_) * np.outer(x, d_sigmoid(z1))  # TODO: Simply by reusing dL1_db1.
        dL2_db2 = 2 * np.linalg.norm(h2 - h2_) * d_sigmoid(z2)
        dL2_dW2 = 2 * np.linalg.norm(h2 - h2_) * np.outer(h1, d_sigmoid(z2))

        # Compute local gradients for backward parameters.
        dL1_dc2 = 2 * np.linalg.norm(h1 - h1_) * d_sigmoid(z1_)  # TODO: Verify that this is correct.
        dL1_dV2 = 2 * np.linalg.norm(h1 - h1_) * np.outer(h2_, d_sigmoid(z1_))

        # Update parameters.
        self.b1 -= lr_forward * dL1_db1
        self.W1 -= lr_forward * dL1_dW1
        self.b2 -= lr_forward * dL2_db2
        self.W2 -= lr_forward * dL2_dW2
        self.c2 -= lr_backward * dL1_dc2
        self.V2 -= lr_backward * dL1_dV2

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


# def create_batches(data, batch_size):
#     return [data[k:k+batch_size] for k in range(0, len(data), batch_size)]


# def train_epoch(net, training_data, lr, batch_size, shuffle=True):
#     if shuffle:
#         random.shuffle(training_data)
#     correct = 0
#     running_loss = 0
#     for i_batch, batch in enumerate(create_batches(training_data, batch_size)):
#         net.backprop_update(batch, lr)
#
#         true_labels = [y_true.flatten() for x, y_true in batch]
#         # TODO: Check if this works for multi-class.
#         #running_loss += cross_entropy(y_pred, true_label)
#         #if true_label == y_pred.argmax():
#         #    correct += 1
#
#     # for i_sample, (x, y_true) in enumerate(training_data):
#     #     x = x.flatten()
#     #     y_true = y_true.flatten()
#     #     y_pred = net.backprop_update(x, y_true, lr)
#     #
#     #     true_label = y_true.argmax()  # target values in training_data they are one-hot vectors (in validation_data they are labels)
#     #     running_loss += cross_entropy(y_pred, true_label)
#     #     if true_label == y_pred.argmax():
#     #         correct += 1
#     #
#     #     if i_sample % 5000 == 1:
#     #         print('{} / {} samples - loss: {:.6f} - accuracy: {:.1f} %'.format(i_sample, len(training_data), running_loss / i_sample, correct / i_sample * 100))
#         if i_batch % 100 == 1:
#             print(i_batch)
#             #print('{} / {} samples - loss: {:.6f} - accuracy: {:.1f} %'.format(i_sample, len(training_data), running_loss / i_sample, correct / i_sample * 100))
#
#     print('Average:\tLoss: {:.6f}\tAccuracy: {:.1f}%'.format(running_loss / len(training_data), correct / len(training_data) * 100))


def train_epoch(net, training_data, lr, shuffle=True):
    if shuffle:
        random.shuffle(training_data)
    correct = 0
    running_loss = 0
    for i_sample, (x, y_true) in enumerate(training_data):
        x = x.flatten()
        y_true = y_true.flatten()
        y_pred = net.target_prop_update(x, y_true, 3, 3, 3)#lr)

        true_label = y_true.argmax()  # target values in training_data they are one-hot vectors (in validation_data they are labels)
        running_loss += cross_entropy(y_pred, true_label)
        if true_label == y_pred.argmax():
            correct += 1

        if i_sample % 5000 == 1:
            print('{} / {} samples - loss: {:.6f} - accuracy: {:.1f} %'.format(i_sample, len(training_data), running_loss / i_sample, correct / i_sample * 100))

    print('Average:\tLoss: {:.6f}\tAccuracy: {:.1f}%'.format(running_loss / len(training_data), correct / len(training_data) * 100))
