from model.mnist import federatedSGD, CNNMnist, MLP
from model.mnist import localTrainingFederatedSGD
from model.round import Round
from model.consts import MNIST

import numpy as np

import torch

import random
import time

import sys

TRAIN_TIME = 1
DIMENSION  = 28 * 28

def afunc(uid_to_weights, prev_weights):
    return federatedSGD(uid_to_weights, prev_weights)

def dfunc(_input, data_pct=1):
    random.shuffle(_input)
    return _input[:int(len(_input) * data_pct)]

def tfunc(t_data, global_weights):
    return localTrainingFederatedSGD(t_data, global_weights)

def handle_update(x, i):
    pass

def handle_delete(x, i):
    pass

def server_thread(aggregator, log, num_rounds, users, train_qs, weight_qs, update_qs, delete_qs, statistics, data_reserves, train_pct=10, mode=0):
    """
    Workflow: (1) Create and run a round... entails asking users to train and
                  retrieving the result
              (2) Update teh global model with user weights
              (3) Receive user requests for updates and deletes and handle them
    """
    global TRAIN_TIME
    start = time.time()
    print("Starting server thread at: " + str(start))
    print("[server thread] running simulation for " + str(num_rounds) + " iterations")

    # initialize global weights of round -1 to be random
    # log.set_global_checkpoint(-1, np.array([random.randint(1, 10) for _ in range(DIMENSION)]))
    # net_glob = CNNMnist().to(MNIST['device'])
    net_glob = MLP(dim_in=784, dim_hidden=200, dim_out=MNIST["num_classes"]).to(MNIST["device"])

    net_glob.train()
    log.set_global_checkpoint(-1, net_glob)

    rid = 0
    for _ in range(num_rounds):
        print("[server thread] Starting round " + str(rid) + " at time: " + str(time.time() - start))
        # determine number of users to participate in the next round
        r = random.randrange(1, len(users) + 1)

        # set up the round
        new_round = Round(
            rid,                    # round id
            tfunc,                  # training function   => placeholder
            dfunc,                  # data function
            afunc,                  # aggregator function => placeholder
            r                       # number of participating devices
        )

        # tell aggregator to make users train on data
        print("[server thread] calling on users to train...")
        aggregator.basic_train(new_round, train_qs, weight_qs)

        # TODO: handle user requests to updates and deletes => update logger
        for i in range(len(users)):
            _update = update_qs[i].deque()
            _delete = delete_qs[i].deque()

            if _update is not None:
                # handle update
                handle_update(_update, i)

            if _delete is not None:
                # handle delete
                handle_delete(_delete, i)

        print("[server thread] computing accuracy on most recent global weights")

        round_stats = {"guesses": 0, "correct": 0, "guess_to_actual":{str(i): [0 for _ in range(10)] for i in range(10)}}

        samples = data_reserves.sample(n = len(data_reserves) // 4)

        for index, x in samples.iterrows():
            model = log.get_global_checkpoint(rid)

            val = torch.tensor([[x['body'].tolist()]]).float()

            y_pred = model(val).detach().numpy()
            y_pred = [abs(x) for x in y_pred[0]]

            if index % 1000 == 0:
                print("Predicted values:", y_pred)

            prediction = y_pred.index(max(y_pred))
            actual = x['target']

            round_stats["guesses"] += 1
            round_stats["correct"] += 1 if prediction == actual else 0
            round_stats["guess_to_actual"][str(prediction)][actual] += 1

        statistics.append(round_stats)
        rid += 1

    return statistics
