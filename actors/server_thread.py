from model.mnist import federatedSGD, CNNMnist
from model.mnist import localTrainingFederatedSGD
from model.round import Round
from model.consts import MNIST

import numpy as np

import random
import time

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

def server_thread(aggregator, log, _tlength, users, train_qs, weight_qs, update_qs, delete_qs, train_pct=10, mode=0):
    """
    Workflow: (1) Create and run a round... entails asking users to train and
                  retrieving the result
              (2) Update teh global model with user weights
              (3) Receive user requests for updates and deletes and handle them
    """
    global TRAIN_TIME
    start = time.time()
    print("Starting server thread at: " + str(start))
    print("[server thread] running simulation for " + str(_tlength) + " seconds")
    # initialize global weights of round -1 to be random
    # log.set_global_checkpoint(-1, np.array([random.randint(1, 10) for _ in range(DIMENSION)]))
    log.set_global_checkpoint(-1, CNNMnist()).to(MNIST['device'])


    rid = 0
    while time.time() - start < _tlength:
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

        # determine users
        participants = random.sample(users, r)
        pids = [users.index(p) for p in participants]

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

        rid += 1
