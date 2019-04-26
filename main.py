import mnist_loader
from network import NeuralNet, train_epoch, train_mirroring_epoch, evaluate

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = NeuralNet(num_hidden=30)

for epoch in range(2):
   print('sds')
   train_mirroring_epoch(net, len(training_data), lr_backward=0.005, weight_decay_backward=0.2)

# Training loop.
for epoch in range(20):
    print('Epoch', epoch+1)
    train_epoch(net, training_data, lr=0.2)
    evaluate(net, validation_data)
    print('-'*80)
    print()
