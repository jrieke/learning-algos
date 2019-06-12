import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import time
import utils


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
print('Using device', device)


def create_bit_sequences(num_batches, batch_size, num_bits=8, seq_len=10):

    for batch_num in range(num_batches):

        # All batches have the same sequence length
        #seq_len = random.randint(min_len, max_len)
        seq = np.random.binomial(1, 0.5, (seq_len, batch_size, num_bits))
        seq = torch.from_numpy(seq)

        # The input includes an additional channel used for the delimiter
        inp = torch.zeros(seq_len*2 + 1, batch_size, num_bits + 1)
        inp[:seq_len, :, :num_bits] = seq
        inp[seq_len, :, num_bits] = 1.0 # delimiter in our control channel
        outp = seq.clone()

        yield inp.float(), outp.float()


def plot_bit_sequence(inp, target, outp=None):
    nrows = 2 if outp is None else 3
    fig, axes = plt.subplots(nrows=nrows, sharex=True, figsize=(5, nrows*2))

    plt.sca(axes[0])
    inp = np.asarray(inp)
    print(inp.T.shape)
    plt.imshow(inp.T, cmap='Greys')
    plt.title('input')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)

    plt.sca(axes[1])
    target = np.asarray(target)
    print(np.hstack([np.zeros((len(target.T), len(target)+1)), target.T]).shape)
    plt.imshow(np.hstack([np.zeros((len(target.T), len(target)+1)), target.T]), cmap='Greys')
    plt.title('target')
    plt.gca().axes.get_yaxis().set_visible(False)

    if outp is not None:
        plt.gca().axes.get_xaxis().set_visible(False)

        plt.sca(axes[2])
        outp = np.asarray(outp)
        print(outp.T.shape)
        plt.imshow(outp.T, cmap='Greys')
        plt.title('output')
        plt.gca().axes.get_yaxis().set_visible(False)


# TODO: For future refactoring into utils.training_loop method.
# def train_loop(model, train_step, train_loader, val_step, val_loader, num_epochs):
#     for epoch in range(num_epochs):
#         print('Epoch', epoch + 1)
#
#         model.train()
#         averager = utils.Averager()
#         start_time = time.time()
#         for batch, data in enumerate(train_loader):
#             metrics = train_step(data)
#             # TODO: Add metrics to averager.
#             if batch % 1000 == 0:
#                 print(f'Batch {batch} / {len(train_loader)} ({time.time()-start_time:.0f} s) - {averager}')
#         print('Took {:.0f} seconds'.format(time.time() - start_time))
#         print('Train set average:\t', averager)
#
#
#         model.eval()
#         averager = utils.Averager()
#         with torch.no_grad():
#             for batch, data in enumerate(val_loader):
#                 metrics = val_step(data)
#                 # TODO: Add metrics to averager.
#             print('Val set average:\t', averager)
#             print('-' * 80)
#             print()
#
#
# def train_copy_task(model, device, params):
#     """Train the network completely."""
#     loss_func = nn.BCEWithLogitsLoss()  # TODO: Check that this operates on the right dimension.
#     optimizer = optim.RMSprop(model.parameters(), lr=params['lr'])
#
#     def train_step(data):
#         input, target = data
#         input, target = input.to(device), target.to(device)
#         output = model(input)
#         targeted_output = output[-len(target):]
#         loss = loss_func(targeted_output, target)
#         acc = (torch.round(torch.sigmoid(targeted_output)) == target).detach().numpy().mean()
#         model.zero_grad()
#         loss.backward()
#         optimizer.step()
#         return {'loss': loss.item(), 'acc': acc}
#
#     def val_step(data):
#         input, target = data
#         input, target = input.to(device), target.to(device)
#         output = model(input)
#         targeted_output = output[-len(target):]
#         loss = loss_func(targeted_output, target)
#         acc = (torch.round(torch.sigmoid(targeted_output)) == target).detach().numpy().mean()
#         return {'loss': loss.item(), 'acc': acc}
#
#     # TODO: Maybe this works directly with generator.
#     utils.train_loop(model, train_step, train_loader, val_step, val_loader)




def train_copy_task(model, params):
    """Train the network completely."""
    loss_func = nn.BCEWithLogitsLoss()  # TODO: Check that this operates on the right dimension.
    optimizer = optim.RMSprop(model.parameters(), lr=params['lr'], momentum=params['momentum'])

    for epoch in range(params['num_epochs']):
        print('Epoch', epoch + 1)

        # --------------------- Train phase -----------------------------
        model.train()
        averager = utils.Averager()
        start_time = time.time()
        for batch, (input, target) in enumerate(create_bit_sequences(params['num_batches'], params['batch_size'], seq_len=params['seq_len'])):

            # Sanity check 1: Train on 0 inputs.
            # if batch == 0: print('WARNING: Sanity check enabled')
            # input = torch.zeros_like(input)

            # Sanity check 2: Train on 0 targets.
            # if batch == 0: print('WARNING: Sanity check enabled')
            # target = torch.zeros_like(target)

            # Sanity check 3: Overfit on a single batch.
            # if batch == 0: print('WARNING: Sanity check enabled')
            # if epoch == 0 and batch == 0:
            #    fixed_batch_data = input, target
            # else:
            #    input, target = fixed_batch_data

            input, target = input.to(device), target.to(device)
            output = model(input)
            targeted_output = output[-len(target):]
            loss = loss_func(targeted_output, target)
            averager.add('loss', loss.item())
            acc = (torch.round(torch.sigmoid(targeted_output)) == target).float().mean().item()
            averager.add('acc', acc)
            model.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 1000 == 0:
                print(f'Batch {batch} ({time.time()-start_time:.0f} s) - {averager}')
        print('Took {:.0f} seconds'.format(time.time() - start_time))
        print('Train set average:\t', averager)

        # --------------------- Validation phase -----------------------------
        model.eval()
        averager = utils.Averager()
        with torch.no_grad():
            for batch, (input, target) in enumerate(create_bit_sequences(params['num_batches_eval'], params['batch_size_eval'], seq_len=params['seq_len'])):
                input, target = input.to(device), target.to(device)
                output = model(input)
                targeted_output = output[-len(target):]
                loss = loss_func(targeted_output, target)
                averager.add('loss', loss.item())
                acc = (torch.round(torch.sigmoid(targeted_output)) == target).float().mean().item()
                averager.add('acc', acc)

        print('Val set average:\t', averager)
        print('-' * 80)
        print()



class LSTM(nn.Module):
    """

    Setting from NTM paper for copy task: num_hidden=256, num_layers=3, lr=3e-5, momentum=0.9

    """

    def __init__(self, num_hidden=64, num_layers=1):
        super(LSTM, self).__init__()
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        # TODO: In the NTM paper for the copy task, they use 3 LSTM layers with 256 hidden neurons each, RMSProp with lr 3e-5 and momentum 0.9.
        self.lstm = nn.LSTM(9, num_hidden, num_layers=num_layers)
        self.linear = nn.Linear(num_hidden, 8)

    def forward(self, input):  # shape of input: seq_len, batch_size, input_size (9)
        seq_len, batch_size, _ = input.shape

        # Propagate through LSTM.
        hidden = [torch.randn(self.num_layers, batch_size, self.num_hidden).to(device),
                  torch.randn(self.num_layers, batch_size, self.num_hidden).to(device)]
        out, (hidden, cell) = self.lstm(input, hidden)  # shape of out: seq_len, batch_size, self.num_hidden

        # Reshape out to (seq_len*batch_size, hidden_size), so that the linear layer can be applied to each bit sequence.
        squeezed_out = out.view(seq_len*batch_size, self.num_hidden)  # shape of squeezed_out: seq_len*batch_size, self.num_hidden

        # Decode bits via linear layer and unpack.
        squeezed_bits = self.linear(squeezed_out)  # shape of squeezed_bits: seq_len*batch_size, output_size (8)
        bits = squeezed_bits.view(seq_len, batch_size, -1)  # shape of bits: seq_len, batch_size, output_size (8)

        # No sigmoid here because it is already included in BCEWithLogits loss.
        return bits

# Simple test setting.
# params = {'num_epochs': 1,
#           'num_hidden': 64,
#           'num_layers': 1,
#           'lr': 0.001,
#           'momentum': 0,
#           'seq_len': 5,
#           'batch_size': 128,
#           'batch_size_eval': 128,
#           'num_batches': 10000,
#           'num_batches_eval': 1000}

# Setting from NTM paper for copy task (3 LSTM layers, 256 hidden neurons, lr 3e-5, momentum 0.9.
# TODO: What's their batch size?
# TODO: Rather use the params from the synthetic gradient paper, because their setting is more similar to mine.
# TODO: For how many epochs/samples did they train?
params = {'num_epochs': 3,
          'num_hidden': 256,
          'num_layers': 3,
          'lr': 3e-5,
          'momentum': 0.9,
          'seq_len': 15,
          'batch_size': 128,
          'batch_size_eval': 128,
          'num_batches': 10000,
          'num_batches_eval': 1000}
net = LSTM(num_hidden=params['num_hidden'], num_layers=params['num_layers']).to(device)
train_copy_task(net, params)


batch_inp, batch_target = list(create_bit_sequences(1, 1, seq_len=15))[0]
batch_inp, batch_target = batch_inp.to(device), batch_target.to(device)
batch_outp = torch.sigmoid(net(batch_inp))
plot_bit_sequence(batch_inp[:, 0].cpu(), batch_target[:, 0].cpu(), batch_outp[:, 0].detach().cpu())
plt.colorbar()
plt.show()
