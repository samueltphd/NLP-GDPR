from model.classes import Round

import random
import time

TRAIN_TIME = 1

def dfunc():
    pass

def afunc():
    pass

def server_thread(aggregator, log, _tlength, users, train_qs, weight_qs, update_qs, delete_qs, train_pct=10, mode=0):
    """
    Workflow: (1) Create and run a round... entails asking users to train and
                  retrieving the result
              (2) Receive user requests for updates and deletes and handle them
    """
    global TRAIN_TIME

    start = time.time()
    print("Starting server thread at: " + str(start))

    # set up log information
    log.rounds                    = {}
    log.uid_to_rids               = {i: [] for i in range(len(users))}
    log.rid_to_uids               = {}
    log.uid_to_weights            = {i: [] for i in range(len(users))}
    log.uid_rid_to_weights        = {}
    log.uid_to_user               = {i: users[i] for i in range(len(users))}
    log.rid_to_round              = {}
    log.rid_to_global_checkpoints = {}
    log.next_rid                  = 1

    rid = 0
    while time.time() - start < _tlength:
        print(time.time() - start)
        # determine number of users to participate in the next round
        r = random.randrange(1, len(users) + 1)

        # set up the round
        round = Round(
            rid,                    # round id
            aggregator.basic_train, # training function
            dfunc,                  # data function       => placeholder
            afunc,                  # aggregator function => placeholder
            r                       # number of participating devices
        )

        # add round to the log
        log.rounds[rid] = round
        log.rid_to_round[rid] = round

        # determine users
        participants = random.sample(users, r)
        pids = [users.index(p) for p in participants]

        print("Sending training requests...")
        # send training request to participants
        for pid, p in zip(pids, participants):
            train_qs[pid].enque(round.training_function)

            if pid in log.uid_to_rids.keys():
                log.uid_to_rids[pid].append(rid)
            else:
                log.uid_to_rids[pid] = [rid]

        # allow time for training
        train_start = time.time()

        # update logs with basic info
        log.rid_to_uids[rid] = list(range(len(participants)))

        rid          += 1
        log.next_rid += 1

        # retrieve user weights
        if time.time() - train_start < TRAIN_TIME:
            print("Now we wait...")
            time.sleep(TRAIN_TIME)


        weights = [0 for _ in range(len(users))]
        for pid in pids:
            w = weight_qs[pid].deque()

            if w is not None:
                weights[pid] = w

                log.uid_rid_to_weights[(pid, rid)] = w
                log.uid_rid_to_weights[uid].append(w)

        # TODO: add weights to global model

        # TODO: handle user requests to updates and deletes
