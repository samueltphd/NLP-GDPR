from actors.server_thread import server_thread
from actors.user_thread import user_thread
from model.classes import User, Round, Aggregator, Log
from utils import PCQueue

from datetime import datetime
import pandas as pd

import random
import sys

TEST_LENGTH = 10.

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

data = pd.read_csv("reddit_data.csv")

def update_id(x):
    global num_users

    return hash(x) % num_users

data['id'] = data['id'].apply(update_id)

print(data.columns)

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
    global log, aggregator

    aggregator = Aggregator(log, mode)

def initialize_users():
    global num_users, data, mode, log, aggregator, users

    users = []
    uids = [-1]

    for _ in range(num_users):
        uid = -1
        while uid in uids:
            uid = data.loc[random.randint(0, len(data))]['id']

        uids.append(uid)
        users.append(User(uid, aggregator, log, mode))

def setup_test():
    global data, users

    initialize_log()
    initialize_aggregator()
    initialize_users()

    # populate each user's "data" field with the half of their data
    for u in users:
        uid = u.uid

        all_posts = data.loc[data['id'] == uid]
        sample_posts = all_posts.sample(n = len(all_posts) // 2)

        u.data = [{'id': uid, 'val': x['body'] for x in sample_posts}]

def run_test():
    global log, aggregator, users, TEST_LENGTH

    train_q  = [PCQUeue() for _ in range(num_users)]
    weight_q = [PCQUeue() for _ in range(num_users)]
    update_q = [PCQUeue() for _ in range(num_users)]
    delete_q = [PCQUeue() for _ in range(num_users)]

    user_threads = [threading.Thread(target=user_thread, args=(users[id], id, TEST_LENGTH, train_q, weight_q, update_q, delete_q)) for id in range(num_users)]
    for u in user_threads:
        u.start()

    server_thread = threading.Thread(target=server_thread, args=(aggregator, log, TEST_LENGTH, users, train_q, weight_q, update_q, delete_q))
    server_thread.start()

    for u in user_threads:
        u.join()

    server_thread.join()

def destroy_test():
    global log, aggregator, users

    log        = None
    aggregator = None
    users      = None

def run():
    global num_tests, tests

    for _ in range(num_tests):
        setup_test()
        run_test()
        destroy_test()

        tests += 1


def main():
    global tests, deletes, updates

    if len(sys.argv) < 7 or len(sys.argv) > 7:
        print("python3 tester.py <num_tests> <num_users> <pct_gdpr_users> <pct_delete> <pct_update> <compliance_mode>")
        exit(1)
    run()

    print("Total Operations: " + str(tests))
    print("Total Deletions:  " + str(deletes))
    print("Total Updates:    " + str(updates))

if __name__ == "__main__":
    main()
