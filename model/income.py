import numpy as np
import random
from .consts import INCOME
from sklearn.model_selection import train_test_split

# copy stuff to copy over neural network
from copy import deepcopy

# function to help import the data
def import_census(file_path, addBias=True):
    '''
        Helper function to import the census dataset

        @param:
            train_path: path to census train data + labels
            test_path: path to census test data + labels
        @return:
            X_train: training data inputs
            Y_train: training data labels
            X_test: testing data inputs
            Y_test: testing data labels
    '''
    data = np.genfromtxt(file_path, delimiter=',', skip_header=False)
    X = data[:, :-1]
    Y = data[:, -1].astype(int)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    # adding bias
    if addBias:
        X_train = np.append(X_train, np.ones((len(X_train), 1)), axis=1)
        X_test = np.append(X_test, np.ones((len(X_test), 1)), axis=1)
    return X_train, Y_train, X_test, Y_test

#### federated learning stuff

def federatedAveraging(w):
    # make a dummy variable
    w_cardinality = len(list(w.values())[0])
    w_avg = np.zeros(w_cardinality)
    for uid in w:
        w_avg += w[uid]
    w_avg = np.divide(w_avg, len(w))
    return w_avg


############################# vanilla stuff #############################
def federatedSGD(uid_to_weight, prev_weights, C=1, learning_rate=0.01):
    # aggregation function:
    # - input:   + dict of weights from all the different devices. format: uid (int): weight (list-like)
    #            + a previous global set of weights (list-like) (or CNN)
    # - output:  + the result of aggregation, probably the checkpoint
    #            + a dict that maps from uid (int) to the weight changes function to be happened (that takes in the current set of
    #              weights from the user, and update it to be something)
    #            +-+-+ weight function: a function that takes in a previous set of weights and output a novel set of weights
    new_global_weight = federatedAveraging(uid_to_weight)
    update_dict = {k: lambda x: new_global_weight for k in uid_to_weight}
    return new_global_weight, update_dict


def localTrainingFederatedSGD(data, global_weights):
    # training function:
    # - input: the data that the data function outputs, AND that takes into account existing weights from the user
    # - output: the weights that are going to be sent back to the aggregator
    assert data != None
    weights = np.array(global_weights, copy=True)
    # random shuffle the data
    random.shuffle(data)
    d1, d2 = [], []
    for data_point in data:
        assert type(data_point["val"]) == np.ndarray
        assert type(data_point["target"]) == np.int64 or type(data_point["target"]) == int
        d1.append(data_point["val"])
        d2.append(data_point["target"])
    data = (np.array(d1), np.array(d2))
    local = LocalUpdate(INCOME, data)
    local_contribution, loss = local.train(weights) # global weights, in this case, is a neural net
    return deepcopy(local_contribution), deepcopy(loss)


######### HELPER STUFF FOR LOGISTIC REGRESSION
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


def loss(samples, labels, weights):
    predictions = np.dot(samples, weights)
    softmaxed = softmax(predictions)
    prediction_labels = np.array([1.0 if e > 0.5 else 0.0 for e in softmaxed])
    same_count = 0
    for i, p in enumerate(prediction_labels):
        if int(p) == int(labels[i]): same_count += 1
    return float(len(labels) - same_count) / float(len(labels))



############################# DEEP LEARNING STUFF - UNFINISHED #############################

class LocalUpdate(object):
    def __init__(self, args, dataset):
        self.args = args
        self.dataset_len = len(dataset[0])
        # dataset here has to be a zip of sample and data point - it shall not be torch yet and we convert them to torch
        # constructing batches
        self.dataset = dataset


    def shuffleData(self, construct_batches=True):
        # data should be a tuple:
        samples, labels = np.array(self.dataset[0], copy=True), np.array(self.dataset[1], copy=True)
        assert len(samples) == len(labels)
        ids = np.arange(len(labels))
        shuffled_samples, shuffled_labels = samples[ids], labels[ids]
        if not construct_batches:
            self.dataset = (shuffled_samples, shuffled_labels)
            return [(shuffled_samples, shuffled_labels)]
        batches = []
        for starting_index in range(0, len(labels), self.args['local_bs']):
            ending_index = min(starting_index + self.args['local_bs'], len(labels))
            b1, b2 = shuffled_samples[starting_index:ending_index], shuffled_labels[starting_index:ending_index]
            batches.append((b1, b2))
        self.dataset = (shuffled_samples, shuffled_labels)
        return batches
        

    # net here is the same as the global neural net at that level, the state dict basically
    def train(self, global_weights):
        weights = np.array(global_weights, copy=True)
        prev_loss, curr_loss = 0, 0
        for iter in range(self.args['local_ep']):
            prev_loss = curr_loss
            curr_loss = 0
            batches = self.shuffleData()
            for bid, (samples, labels) in enumerate(batches):
                assert len(samples) == len(labels)
                # predictions
                predictions = np.dot(samples, weights).astype(float)
                softmaxed = softmax(predictions)
                assert len(softmaxed) == len(labels)
                diff = np.subtract(softmaxed, labels)
                dotted = np.dot(samples.transpose(), diff)
                gradient = np.divide(dotted, self.args['local_bs'])
                # update weights
                weights = weights - self.args['lr']*gradient
            curr_loss = loss(self.dataset[0], self.dataset[1], weights)
            print("current loss: ", curr_loss)
            if abs(curr_loss - prev_loss) <= self.args['converge_threshold']:
                break
        print("Training loss: ", curr_loss)
        return weights, curr_loss
