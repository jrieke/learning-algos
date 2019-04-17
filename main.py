import mnist_loader
import numpy as np
import scipy as sp
import scipy.special
import sklearn.metrics
import random

sigmoid = sp.special.expit
softmax = sp.special.softmax

def cross_entropy(y_pred, true_label):
    # TODO: Check if this holds for multiple samples.
    return -np.log(y_pred[true_label])


def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

len(training_data)

W1 = np.random.randn(784, 30)
W2 = np.random.randn(30, 10)
b1 = np.random.rand(30)
b2 = np.random.rand(10)

training_data[0][0].shape, training_data[0][1].shape

x = training_data[0][0].reshape(-1)
y_true = training_data[0][1].reshape(-1)
x.shape, y_true.shape

def forward(x):
    z1 = x @ W1 + b1
    a1 = sigmoid(z1)

    z2 = a1 @ W2 + b2
    a2 = softmax(z2)  # TODO: When using mini batches, do axis=1.

    return z1, a1, z2, a2

[a.shape for a in forward(x)]

def backprop_update(x, y_true, lr):
    global b1, b2, W1, W2  # TODO: Remove once this is in network.
    z1, a1, z2, a2 = forward(x)

    # Compute errors for each layer.
    e2 = a2 - y_true
    e1 = d_sigmoid(z2) * e2 @ W2.T
    #e1.shape, e2.shape

    # Compute gradients of cost w.r.t. parameters.
    grad_b1 = e1
    grad_b2 = e2
    grad_W1 = np.outer(x, e1)  # creates a matrix from two vectors
    grad_W2 = np.outer(a1, e2)
    #grad_W1.shape, grad_W2.shape, grad_b1.shape, grad_b2.shape

    b1 -= lr * grad_b1
    b2 -= lr * grad_b2
    W1 -= lr * grad_W1
    W2 -= lr * grad_W2

    return a2


def evaluate():
    correct = 0
    running_loss = 0
    for i_sample, (x, y_true) in enumerate(validation_data):
        x = x.flatten()
        y_true = y_true.flatten()
        _, _, _, y_pred = forward(x)
        #print(x.shape, y_true, y_pred.shape)

        true_label = y_true[0]  # target values in validation_data are labels (in training_data they are one-hot vectors)
        running_loss += cross_entropy(y_pred, true_label)
        if true_label == y_pred.argmax():
            correct += 1

    avg_loss = running_loss / len(validation_data)
    avg_acc = correct / len(validation_data) * 100

    print('Evaluating:\tLoss: {:.4f}\tAccuracy: {:.1f}%\n'.format(avg_loss, avg_acc))

evaluate()

def train_epoch(lr, shuffle=True):
    if shuffle:
        random.shuffle(training_data)
    correct = 0
    running_loss = 0
    for i_sample, (x, y_true) in enumerate(training_data):
        x = x.flatten()
        y_true = y_true.flatten()
        y_pred = backprop_update(x, y_true, lr)

        true_label = y_true.argmax()  # target values in training_data they are one-hot vectors (in validation_data they are labels)
        running_loss += cross_entropy(y_pred, true_label)
        if y_true.argmax() == y_pred.argmax():
            correct += 1

        if i_sample % 5000 == 1:
            print('{} / {} samples - loss: {:.6f} - accuracy: {:.1f} %'.format(i_sample, len(training_data), running_loss / i_sample, correct / i_sample * 100))

    print('Average:\tLoss: {:.6f}\tAccuracy: {:.1f}%'.format(running_loss / len(training_data), correct / len(training_data) * 100))

# Training loop.
for epoch in range(1, 10):
    print('Epoch', epoch)
    train_epoch(lr=0.005)
    evaluate()
    print('-'*80)
    print()



# TODO: Implement mini-batching.
def create_batches(data, batch_size, shuffle=True):
    if shuffle:
        random.shuffle(data)
    return [data[k:k+batch_size] for k in range(0, len(data), batch_size)]

#create_batches(training_data, 10)
