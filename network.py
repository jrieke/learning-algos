import numpy as np
import scipy as sp
import scipy.special
import random


sigmoid = sp.special.expit
softmax = sp.special.softmax


def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def cross_entropy(y_pred, true_label):
    # TODO: Check if this holds for multiple samples.
    return -np.log(y_pred[true_label])


class NeuralNet:

    def __init__(self, num_hidden=30):
        self.W1 = np.random.randn(784, num_hidden)
        self.W2 = np.random.randn(num_hidden, 10)
        self.b1 = np.random.rand(num_hidden)
        self.b2 = np.random.rand(10)

        # Fixed random weights for Feedback Alignment.
        self.B2 = np.random.randn(num_hidden, 10)

    def forward(self, x):
        # Hidden layer.
        z1 = x @ self.W1 + self.b1
        a1 = sigmoid(z1)

        # Output layer.
        z2 = a1 @ self.W2 + self.b2
        # TODO: Using sigmoid here improves this a lot.
        a2 = sigmoid(z2)  # TODO: When using mini batches, do axis=1.

        return z1, a1, z2, a2

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
        z1, a1, z2, a2 = self.forward(x)

        # Compute errors for each layer (= gradient of cost w.r.t layer input).
        e2 = d_sigmoid(z2) * (a2 - y_true)  # gradient through cross entropy and softmax
        e1 = d_sigmoid(z1) * (e2 @ self.W2.T)  # gradient backpropagation

        # Compute gradients of cost w.r.t. parameters.
        grad_b1 = e1
        grad_b2 = e2
        grad_W1 = np.outer(x, e1)  # np.outer creates a matrix from two vectors
        grad_W2 = np.outer(a1, e2)

        # Update parameters.
        self.b1 -= lr * grad_b1
        self.b2 -= lr * grad_b2
        self.W1 -= lr * grad_W1
        self.W2 -= lr * grad_W2

        return a2

    def final_layer_update(self, x, y_true, lr):
        # Run forward pass.
        z1, a1, z2, a2 = self.forward(x)

        # Compute errors for each layer (= gradient of cost w.r.t layer input).
        e2 = d_sigmoid(z2) * (a2 - y_true)  # gradient through cross entropy and softmax

        # Compute gradients of cost w.r.t. parameters.
        grad_b2 = e2
        grad_W2 = np.outer(a1, e2)

        # Update parameters.
        self.b2 -= lr * grad_b2
        self.W2 -= lr * grad_W2

        return a2

    def feedback_alignment_update(self, x, y_true, lr):
        # Run forward pass.
        z1, a1, z2, a2 = self.forward(x)

        # Compute errors for each layer (= gradient of cost w.r.t layer input).
        e2 = d_sigmoid(z2) * (a2 - y_true)  # gradient through cross entropy and softmax
        e1 = d_sigmoid(z1) * (e2 @ self.B2.T)  # gradient backpropagation

        # Compute gradients of cost w.r.t. parameters.
        grad_b1 = e1
        grad_b2 = e2
        grad_W1 = np.outer(x, e1)  # np.outer creates a matrix from two vectors
        grad_W2 = np.outer(a1, e2)

        # Update parameters.
        self.b1 -= lr * grad_b1
        self.b2 -= lr * grad_b2
        self.W1 -= lr * grad_W1
        self.W2 -= lr * grad_W2

        return a2

    def weight_mirroring_update(self, x, y_true, lr):
        pass

    def target_prop_update(self, x, y_true, lr):
        pass

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
        y_pred = net.backprop_update(x, y_true, lr)

        true_label = y_true.argmax()  # target values in training_data they are one-hot vectors (in validation_data they are labels)
        running_loss += cross_entropy(y_pred, true_label)
        if true_label == y_pred.argmax():
            correct += 1

        if i_sample % 5000 == 1:
            print('{} / {} samples - loss: {:.6f} - accuracy: {:.1f} %'.format(i_sample, len(training_data), running_loss / i_sample, correct / i_sample * 100))

    print('Average:\tLoss: {:.6f}\tAccuracy: {:.1f}%'.format(running_loss / len(training_data), correct / len(training_data) * 100))
