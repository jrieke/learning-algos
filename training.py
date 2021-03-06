import numpy as np
import random
import mnist_loader
import collections
import utils
import time


def cross_entropy(y_pred, true_label):
    return -np.log(y_pred[true_label])


def evaluate(net, evaluation_data, params):
    """Evaluate the network on the evaluation_data."""
    averager = utils.Averager()
    for i_sample, (x, y_true) in enumerate(evaluation_data):
        x = x.flatten()
        y_true = y_true.flatten()
        y_pred = net.forward(x, params)

        true_label = y_true[0]  # target values in validation_data are labels (in training_data they are one-hot vectors)
        averager.add('loss', cross_entropy(y_pred, true_label))
        averager.add('acc', true_label == y_pred.argmax())

    print('Eval set average:\t', averager)


def train_epoch(net, training_data, params):
    """Train the network for one epoch on the training_data."""
    random.shuffle(training_data)
    averager = utils.Averager()
    start_time = time.time()
    for i_sample, (x, y_true) in enumerate(training_data):
        x = x.flatten()
        y_true = y_true.flatten()

        y_pred = net.update(x, y_true, params, averager)

        true_label = y_true.argmax()  # target values in training_data they are one-hot vectors (in validation_data they are labels)
        averager.add('loss', cross_entropy(y_pred, true_label))
        averager.add('acc', true_label == y_pred.argmax())

        if i_sample % 1000 == 0:
            print(f'{i_sample} / {len(training_data)} samples ({time.time()-start_time:.0f} s) - {averager}')

    print('Took {:.0f} seconds'.format(time.time()-start_time))
    print('Train set average:\t', averager)


def train(net, params, num_epochs=20):
    """Train the network completely."""
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    for epoch in range(num_epochs):
        print('Epoch', epoch + 1)
        train_epoch(net, training_data, params)
        evaluate(net, validation_data, params)
        print('-' * 80)
        print()


def train_mirroring(net, params, num_epochs=2):
    """Train only the mirroring phase of a WeightMirroringNetwork (50k iterations per epoch, i.e. the same as MNIST)."""
    print()
    for epoch in range(num_epochs):
        angle_before = np.rad2deg(utils.angle_between(net.V2.flatten(), net.W2.T.flatten()))
        mean_before = net.V2.mean()
        for i_sample in range(50000):  # as many samples as in MNIST training set
            net.update_weight_mirror(params)
        angle_after = np.rad2deg(utils.angle_between(net.V2.flatten(), net.W2.T.flatten()))
        mean_after = net.V2.mean()
        print(f'Ran mirroring for one epoch, angle: {angle_before:.3f}° -> {angle_after:.3f}°, mean of backward weight: {mean_before:.3f} -> {mean_after:.3f}')
    print()
