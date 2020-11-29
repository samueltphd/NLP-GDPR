import numpy as np
import random

# Neural network stuff
import torch
from torch import nn
from .consts import MNIST
import torch.nn.functional as F

# copy stuff to copy over neural network
from copy import deepcopy

# import sys

def federatedAveraging(w):
    # make a dummy variable
    w_avg = deepcopy(list(w.values())[0])
    for k in w_avg.keys():
        # for each uid
        for uid in w:
            w_avg[k] += w[uid][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


############################# vanilla stuff #############################
def federatedSGD(uid_to_weight, prev_weights, C=1, learning_rate=0.01):
    # aggregation function:
    # - input:
    #       + dict of weights from all the different devices. format:
    #               uid (int): weight (list-like)
    #       + a previous global set of weights (list-like) (or CNN)
    #
    # - output:
    #       + the result of aggregation, probably the checkpoint
    #       + a dict that maps from uid (int) to the weight changes function to be happened (that takes in
    #           the current set of weights from the user, and update it to be something)
    #           +-+-+ weight function: a function that takes in a previous set of weights and output a novel set of weights
    new_global_weight = federatedAveraging(uid_to_weight)
    # copy the previous weights to the new one
    new_weights = deepcopy(prev_weights)
    # and then copy aggregated to the new one
    new_weights.load_state_dict(new_global_weight)
    # for this basic learning algorithm, just ask everyone to update their local stuff to
    # be the neural net
    update_dict = {k: lambda x: new_weights for k in uid_to_weight}
    return new_weights, update_dict


def localTrainingFederatedSGD(data, global_weights):
    # training function:
    # - input: the data that the data function outputs, AND that takes into account existing weights from the user
    # - output: the weights that are going to be sent back to the aggregator
    # random shuffle the data
    assert data != None
    # random shuffle the data
    random.shuffle(data)
    d1, d2 = [], []
    for data_point in data:
        d1.append([data_point["val"]])
        d2.append(data_point["target"])
    data = (d1, d2)
    local = LocalUpdate(MNIST, data)
    state_dict, loss = local.train(deepcopy(global_weights)) # global weights, in this case, is a neural net
    return deepcopy(state_dict), deepcopy(loss)


############################# DEEP LEARNING STUFF - UNFINISHED #############################

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, 5)
        self.fc1 = nn.Linear(576, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)


    def forward(self, x): #x: [10, 1, 28, 28]
        x = F.relu(self.conv1(x))
        x = x.view(-1, 576)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LocalUpdate(object):
    def __init__(self, args, dataset):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.dataset_len = len(dataset[0])
        # dataset here has to be a zip of sample and data point - it shall not be torch yet and we convert them to torch
        # constructing batches
        batches = []
        for index in range(0, len(dataset[0]), self.args['local_bs']):
            ending_index = min(index+self.args['local_bs'], len(dataset[0]))
            b1 = torch.tensor(dataset[0][index:ending_index], dtype=torch.float32)
            b2 = torch.tensor(dataset[1][index:ending_index]).long()
            batches.append((b1, b2))
        self.ldr_train = batches

    # net here is the same as the global neural net at that level, the state dict basically
    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args['lr'], momentum=self.args['momentum'])
        epoch_loss = []
        for iter in range(self.args['local_ep']):
            print("Iteration:", iter)
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args['device']), labels.to(self.args['device'])
                optimizer.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                prediction = [k.tolist().index(max(k)) for k in log_probs]
                loss.backward()
                optimizer.step()
                # if self.args['verbose'] and batch_idx % 10 == 0:
                    # print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    #     iter, batch_idx * len(images), self.dataset_len,
                    #            100. * batch_idx / len(images), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            if len(epoch_loss) >= 2 and abs(epoch_loss[-1] - epoch_loss[-2]) <= self.args['converge_threshold']:
                break
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
