from model.mnist import federatedSGD, MLP
from model.mnist import localTrainingFederatedSGD
from model.round import Round
from model.consts import MNIST

from torch import nn

import numpy as np

import torch

import random
import time

import sys

TRAIN_TIME = 100
DIMENSION  = 28 * 28

def afunc(uid_to_weights, prev_weights):
    return federatedSGD(uid_to_weights, prev_weights)

def dfunc(_input, data_pct=0.25):
    lst = _input[:]
    random.shuffle(lst)
    return lst[:int(len(lst) * data_pct)]

def tfunc(t_data, global_weights):
    return localTrainingFederatedSGD(t_data, global_weights)

def handle_update(x, i):
    pass

def handle_delete(x, i):
    pass

def server_thread(aggregator, log, users, train_qs, weight_qs, statistics, xtest, ytest, train_pct=10, mode=0):
    """
    Workflow: (1) Create and run a round... entails asking users to train and
                  retrieving the result
              (2) Update the global model with user weights
              (3) Receive user requests for updates and deletes and handle them
    """
    start = time.time()
    print("Starting server thread at: " + str(start))

    # initialize global weights of round -1 to be random
    # log.set_global_checkpoint(-1, np.array([random.randint(1, 10) for _ in range(DIMENSION)]))
    net_glob = MLP(dim_in=784, dim_hidden=200, dim_out=MNIST["num_classes"]).to(MNIST["device"])

    net_glob.train()
    log.set_global_checkpoint(-1, net_glob)

    rid = 0
    for _ in range(len(users)):
        print("[server thread] Starting round " + str(rid) + " at time: " + str(time.time() - start))
        # determine number of users to participate in the next round
        r = random.randrange(1, len(users) + 1)

        # set up the round
        new_round = Round(
            rid,                    # round id
            tfunc,                  # training function
            dfunc,                  # data function
            afunc,                  # aggregator function
            r                       # number of participating devices
        )

        # tell aggregator to make users train on data
        print("[server thread] calling on users to train...")
        if not aggregator.basic_train(new_round, train_qs, weight_qs, TRAIN_TIME):
            continue

        print("[server thread] handling user update requests...")
        handled = aggregator.urm.handle_requests()

        print("[server thread] computing accuracy on most recent global weights")

        round_stats = {"guesses": 0, "correct": 0, "guess_to_actual":{str(i): [0 for _ in range(10)] for i in range(10)}}
        round_stats["round"] = rid

        indices = random.sample(list(range(len(xtest))), len(xtest) // 4)

        images, labels = [], []
        for i in indices:
            images.append([xtest[i]])
            labels.append(ytest[i])
        images, labels = torch.tensor(images).float(), torch.tensor(labels)
        model = log.get_global_checkpoint(rid)
        predictions_probs = model(images)
        print("[server thread] Finished predicting, now calculating the actual value")
        prediction = [k.tolist().index(max(k)) for k in predictions_probs]
        loss_f = nn.CrossEntropyLoss()
        loss = loss_f(predictions_probs, labels).item()

        for pred, act in zip(prediction, labels):
            round_stats["guesses"] += 1
            round_stats["correct"] += 1 if pred == act else 0
            round_stats["guess_to_actual"][str(pred)][act] += 1

        round_stats['time'] = time.time() - start

        if handled:
            round_stats['requests_handled'] = 1
        else:
            round_stats['requests_handled'] = 0

        round_stats['logger_size'] = log.getsize()

        print("Round stats: ", round_stats)
        statistics.append(round_stats)

        rid += 1

    return statistics
