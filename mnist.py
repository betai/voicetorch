from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import scipy.misc
import numpy as np
import soundfile as sf
import os

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

import os

all_audio_files = []
def audio_files(dir):
    global all_audio_files
    if len(all_audio_files) > 0:
        return all_audio_files
    all_audio_files = flatten(audio_files_helper(dir))
    return all_audio_files

def flatten(arr):
    result = []
    for elem in arr:
        if isinstance(elem, str) and elem.endswith('.flac'):
            result.append(elem)
        elif isinstance(elem, str):
            continue
        else:
            result.extend(flatten(elem))
    return result

BLOCK_SIZE=256
WINDOW_SIZE=BLOCK_SIZE*2

def audio_files_helper(dir):
    return [audio_files_helper(os.path.join(dir, elem)) if os.path.isdir(os.path.join(dir, elem)) else os.path.join(dir, elem) for elem in os.listdir(dir)]

def random_samples():
    files = audio_files('/Users/betai/Work/voicetorch/LibriSpeech')
    for audio_file in files:
        try:
            current_file_blocks = sf.blocks(audio_file, blocksize=BLOCK_SIZE, overlap=0)
            for block in current_file_blocks:
                if len(block) == BLOCK_SIZE:
                    fft_data = np.fft.fft(block)
                    yield np.concatenate([np.real(fft_data), np.imag(fft_data)])
        except:
            continue

def minibatches():
    batch = []
    for block in random_samples():
        batch.append(block)
        if len(batch) == args.batch_size:
            yield np.array(batch, dtype=np.float32)
            batch = []

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(WINDOW_SIZE, 256)
        self.fc2 = nn.Linear(256, 64)
        self.rfc1 = nn.Linear(64, 256)
        self.rfc2 = nn.Linear(256, WINDOW_SIZE)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        # x == Embedding
        x = F.relu(self.rfc1(x))
        x = F.dropout(x, training=self.training)
        x = self.rfc2(x)
        return x

model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

def save_image(filename, v):
    arr = np.asarray(v.data.numpy())
    scipy.misc.imsave(filename, arr)

saved_image_count = 0

def train(epoch):
    global saved_image_count
    model.train()
    for batch_idx, data in enumerate(minibatches()):
        if args.cuda:
            data = data.cuda()
        data = Variable(torch.from_numpy(data))
        optimizer.zero_grad()
        output = model(data)
        loss = torch.sum((output - data) ** 2)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data),
                loss.data[0]))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    #test()
