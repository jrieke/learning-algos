import mnist_loader
from network import NeuralNet, train_epoch, evaluate

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = NeuralNet(num_hidden=30)

# Training loop.
for epoch in range(1, 20):
    print('Epoch', epoch)
    train_epoch(net, training_data, lr=0.2)
    evaluate(net, validation_data)
    print('-'*80)
    print()
