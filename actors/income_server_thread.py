from model.income import federatedSGD
from model.income import localTrainingFederatedSGD
from model.income import softmax
from model.income import loss
from model.round import Round
from model.consts import INCOME

import numpy as np

import random
import time

import sys

TRAIN_TIME = 1
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
              (2) Update teh global model with user weights
              (3) Receive user requests for updates and deletes and handle them
    """
    global TRAIN_TIME
    start = time.time()
    print("Starting server thread at: " + str(start))

    # initialize global weights of round -1 to be random
    initial_weights = np.zeros(xtest[0].shape[0])
    log.set_global_checkpoint(-1, initial_weights)

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

        print("[server thread] computing accuracy on most recent global weights")

        # round_stats = {"guesses": 0, "correct": 0, "guess_to_actual":{str(i): [0 for _ in range(10)] for i in range(10)}}
        # round_stats["round"] = rid

        indices = random.sample(list(range(len(xtest))), len(xtest) // 4)

        samples, labels = [], []
        for i in indices:
            samples.append(xtest[i])
            labels.append(ytest[i])
        samples, labels = np.array(samples).astype(float), np.array(labels)
        model_weights = log.get_global_checkpoint(rid)
        # and then this part is to create a dummy LocalUpdate 
        prediction_loss = loss(samples, labels, model_weights)
        print("Loss: ", prediction_loss, ", at round: ", rid)


        print("[server thread] handling user update requests...")
        aggregator.urm.handle_requests()

        rid += 1

    print("[server thread] handling user update requests...")
    # aggregator.urm.handle_requests(batch_size=0)
    aggregator.urm.handle_requests()

    return statistics
