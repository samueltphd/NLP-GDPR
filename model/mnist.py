import numpy as np
import random

# Neural network stuff
import torch
from torch import nn
import torch.nn.functional as F

############################# vanilla stuff #############################

def FederatedSGD(uid_to_weight, prev_weights, C=1, learning_rate=0.01):
    # aggregation function:
    # - input:
    #       + dict of weights from all the different devices. format:
    #               uid (int): weight (list-like)
    #       + a previous global set of weights (list-like)
    #
    # - output:
    #       + the result of aggregation, probably the checkpoint
    #       + a dict that maps from uid (int) to the weight changes function to be happened (that takes in
    #           the current set of weights from the user, and update it to be something)
    #           +-+-+ weight function: a function that takes in a previous set of weights and output a novel set of weights
    running_sum = None
    for uid in uid_to_weight:
        weight = uid_to_weight[uid]
        if running_sum == None: running_sum = weight.copy()
        else:
            assert len(weight) == len(running_sum)
            running_sum += weight
    gradient = learning_rate*running_sum
    assert len(gradient) == len(prev_weights)
    new_weight = prev_weights - gradient
    # for this basic learning algorithm, just ask everyone to update to the same weight
    update_dict = {k: lambda x: new_weight for k in uid}
    return new_weight, update_dict


def localTrainingFederatedSGD(data, global_weights, num_batches=1, epochs=100, learning_rate=0.01):
    # training function:
    # - input: the data that the data function outputs, AND that takes into account existing weights from the user
    # - output: the weights that are going to be sent back to the aggregator
    # random shuffle the data
    assert data != None
    # random shuffle the data
    random.shuffle(data)
    Xs, Ys = np.array([data_point["x"] for data_point in data]), np.array([data_point["y"] for data_point in data])
    data = zip(Xs, Ys)
    # determine what the number of batches is
    batches = np.split(data, num_batches)
    new_weights = global_weights.copy()
    for ep in epochs:
        for batch in batches:
            Xs, Ys = np.array([np.array(ele[0]) for ele in batch]), np.array([ele[1] for ele in batch])
            prediction = np.dot(global_weights, Xs) # maybe factor in the bias factor?
            loss = prediction-Ys
            new_weights = new_weights-learning_rate*loss
    return new_weights


############################# DEEP LEARNING STUFF - UNFINISHED #############################


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


