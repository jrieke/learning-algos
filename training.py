import numpy as np
import random
import mnist_loader


def cross_entropy(y_pred, true_label):
    # TODO: Check if this holds for multiple samples.
    return -np.log(y_pred[true_label])


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'."""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


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


def train_epoch(net, training_data, params):
    random.shuffle(training_data)
    correct = 0
    running_loss = 0
    for i_sample, (x, y_true) in enumerate(training_data):
        x = x.flatten()
        y_true = y_true.flatten()
        y_pred = net.update(x, y_true, params)

        true_label = y_true.argmax()  # target values in training_data they are one-hot vectors (in validation_data they are labels)
        running_loss += cross_entropy(y_pred, true_label)
        if true_label == y_pred.argmax():
            correct += 1

        if i_sample % 5000 == 1:
            print('{} / {} samples - loss: {:.6f} - accuracy: {:.1f} %'.format(i_sample, len(training_data), running_loss / i_sample, correct / i_sample * 100))

    print('Average:\tLoss: {:.6f}\tAccuracy: {:.1f}%'.format(running_loss / len(training_data), correct / len(training_data) * 100))

    # TODO: Print this only for nets that have an explicit feedback connection.
    #print('Angle between V2 and W2.T', np.rad2deg(angle_between(net.V2.flatten(), net.W2.T.flatten())))
    #print('Mean of V2 ', net.V2.mean())


def train(net, params, num_epochs=20):
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    for epoch in range(num_epochs):
        print('Epoch', epoch + 1)
        train_epoch(net, training_data, params)
        evaluate(net, validation_data)
        print('-' * 80)
        print()


def train_mirroring(net, params, num_epochs=2):
    for epoch in range(num_epochs):
        angle_before = np.rad2deg(angle_between(net.V2.flatten(), net.W2.T.flatten()))
        mean_before = net.V2.mean()
        for i_sample in range(50000):  # as many samples as in MNIST training set
            net.update_weight_mirror(params)
        angle_after = np.rad2deg(angle_between(net.V2.flatten(), net.W2.T.flatten()))
        mean_after = net.V2.mean()
        # TODO: Somehow this doesn't print properly. When run twice, it only prints the second line.
        print(f'Ran mirroring for one epoch, reduced angle between V2 and W2.T from {angle_before}° to {angle_after}°, mean of V2 changed from {mean_before} to {mean_after}')

