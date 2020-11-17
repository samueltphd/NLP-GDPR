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
    return new_weight, update_dict


def localTrainingFederatedSGD(data, global_weights):
    # training function:
    # - input: the data that the data function outputs, AND that takes into account existing weights from the user
    # - output: the weights that are going to be sent back to the aggregator
    # random shuffle the data
    assert data != None
    # random shuffle the data
    random.shuffle(data)
    data = [(data_point["val"], data_point["target"]) for data_point in data]
    local = LocalUpdate(MNIST, data)
    state_dict, loss = local.train(deepcopy(global_weights)) # global weights, in this case, is a neural net
    return deepcopy(state_dict), deepcopy(loss)


############################# DEEP LEARNING STUFF - UNFINISHED #############################


class CNNMnist(nn.Module):
    def __init__(self, args=MNIST):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(MNIST['num_channels'], 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, MNIST['num_classes'])

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class LocalUpdate(object):
    def __init__(self, args, dataset):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        # dataset here has to be a zip of sample and data point - it shall not be torch yet and we convert them to torch
        # constructing batches
        batches = []
        for index in range(0, len(dataset), self.args['local_bs']):
            batch = dataset[index:min(index+self.args['local_bs'], len(dataset))]
            batches.append(batch)
        self.ldr_train = batches

    # net here is the same as the global neural net at that level, the state dict basically
    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args['lr'], momentum=self.args['momentum'])
        epoch_loss = []
        for iter in range(self.args['local_ep']):
            # print("Iteration:", iter)
            batch_loss = []
            for batch_idx, lst in enumerate(self.ldr_train):
                images, labels = torch.tensor([x[0] for x in lst]), torch.tensor([x[1] for x in lst])
                # print("batch id: ", batch_idx)
                images, labels = images.to(self.args['device']), labels.to(self.args['device'])
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args['verbose'] and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        # print("returning: ", net.state_dict(), sum(epoch_loss) / len(epoch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
