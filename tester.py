from actors.server_thread import server_thread
from actors.user_thread import user_thread

from model.aggregator import Aggregator
from model.logger import Log
from model.round import Round
from model.user import User

from utils import PCQueue

from tensorflow.keras.datasets import mnist

from datetime import datetime
import pandas as pd

import random
import sys
import threading

# trying out with new mnist dataset
from torchvision import datasets, transforms

try:
    num_tests        = int(sys.argv[1])
    num_users        = int(sys.argv[2])
    pct_gdpr_users   = int(sys.argv[3])
    pct_delete       = int(sys.argv[4])
    pct_update       = int(sys.argv[5])
    mode             = int(sys.argv[6])
except Exception:
    print("python3 tester.py <num_tests> <num_users> <pct_gdpr_users> <pct_delete> <pct_update> <compliance_mode>")
    exit(1)

trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
xtrain, ytrain, xtest, ytest = dataset_train.data.tolist(), dataset_train.targets.tolist(), dataset_test.data.tolist(), dataset_test.targets.tolist()


data = pd.DataFrame(columns=['id','body','target'], data=[[random.randint(1,100), x, y] for x, y in zip(xtrain, ytrain)])

def update_id(x):
    global num_users

    return hash(x) % num_users

data['id'] = data['id'].apply(update_id)

data, data_reserves = data.iloc[:, :len(data) // 2], data.iloc[:len(data) // 2, :]

tests      = 0
deletes    = 0
updates    = 0

log        = None
aggregator = None
users      = None

def initialize_log():
    global log

    log = Log()

def initialize_aggregator():
    global log, aggregator, mode

    aggregator = Aggregator(log, mode)

def initialize_users():
    global num_users, data, mode, log, aggregator, users

    users = []
    uids = [-1]

    for _ in range(num_users):
        uid = -1
        while uid in uids:
            uid = data.loc[random.randint(0, len(data) - 1)]['id'] % num_users

        uids.append(uid)
        u = User(uid, aggregator, log, mode)
        users.append(u)

        log.add_user(uid, u)

def setup_test():
    global data, users

    print("Setting up environment...")

    initialize_log()
    initialize_aggregator()
    initialize_users()

    # populate each user's "data" field with the half of their data
    for u in users:
        uid = u.uid

        all_posts = data.loc[data['id'] == uid]
        sample_posts = all_posts.sample(n = len(all_posts) // 2)

        i = 0
        for index, x in sample_posts.iterrows():
            u.add_data({'id': i, 'val': x['body'], 'target': x['target']})
            i += 1

    print("All data flushed!")

def run_test():
    global log, aggregator, users, data_reserves, pct_delete, pct_update

    print("Running test...")

    train_q  = [PCQueue() for _ in range(num_users)]
    weight_q = [PCQueue() for _ in range(num_users)]
    stop_q   = [PCQueue() for _ in range(num_users)]

    statistics = []

    user_threads = [threading.Thread(target=user_thread, args=(users[id], aggregator, log, train_q[id], weight_q[id], stop_q[id], data_reserves, pct_delete, pct_update)) for id in range(num_users)]
    for u in user_threads:
        u.start()

    st = threading.Thread(target=server_thread, args=(aggregator, log, users, train_q, weight_q, statistics, xtest, ytest))
    st.start()

    st.join()

    for q in stop_q:
        q.enque(True)

    for u in user_threads:
        u.join()

    return statistics

def destroy_test():
    global log, aggregator, users

    log        = None
    aggregator = None
    users      = None

def run():
    global num_tests, tests

    results = []
    for _ in range(num_tests):
        setup_test()
        results.append(run_test())
        destroy_test()

        tests += 1

    return results


def main():
    global tests, deletes, updates

    if len(sys.argv) < 7 or len(sys.argv) > 7:
        print("python3 tester.py <num_tests> <num_users> <pct_gdpr_users> <pct_delete> <pct_update> <compliance_mode>")
        exit(1)

    results = run()

    f = open("-".join(sys.argv[2:]) + ".csv", "w")
    f.write("round,time,space,requests_handled,guesses,correct," + ",".join([",".join([str(x) + "-to-" + str(y) for y in range(10)]) for x in range(10)]) + '\n')

    for test in results:
        for r in test:
            tokens = ','.join([','.join([str(x) for x in r['guess_to_actual'][str(i)]]) for i in range(10)])
            f.write(str(r['round']) + ',' + str(r['time']) + ',' + str(r['logger_size']) + ',' + str(r['requests_handled']) + ',' + str(r['guesses']) + ',' + str(r['correct']) + ',' + tokens + '\n')
    f.close()

    print("Test complete. Exiting.")

if __name__ == "__main__":
    main()
