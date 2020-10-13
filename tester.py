from datetime import datetime
import pandas as pd

import random
import sys

try:
    num_loops        = int(sys.argv[1])
    num_interactions = int(sys.argv[2])
    pct_gdpr_users   = int(sys.argv[3])
    pct_delete       = int(sys.argv[4])
    pct_update       = int(sys.argv[5])
except Exception:
    print("python3 tester.py <num_loops> <num_interactions> <pct_gdpr_users> <pct_delete> <pct_update>")
    exit(1)

data = pd.read_csv("reddit_data.csv")

data, data_reserves = data.iloc[:, :len(data) // 2], data.iloc[:len(data) // 2, :]

operations = 0
deletes    = 0
updates    = 0

def delete_data():
    global data, deletes

    # instantaneous deletion
    while True:
        r = random.randrange(len(data))
        try:
            data = data.drop(index = r)
            break
        except KeyError:
            continue
    deletes += 1
    pass

def update_data():
    global data, data_reserves, updates

    # instantaneous update
    r1 = random.randrange(len(data))
    r2 = random.randrange(len(data_reserves))

    temp                   = data.iloc[r1]
    data.iloc[r1]          = data_reserves.iloc[r2]
    data_reserves.iloc[r2] = temp

    updates += 1
    pass

def do_nothing():
    pass

def main_loop():
    global operations

    start = datetime.now()

    for _ in range(num_interactions):
        r = random.randrange(100)

        if r < pct_gdpr_users:
            r = random.randrange(100)

            if r < pct_delete:
                delete_data()
                pass
            elif r < pct_delete + pct_update:
                update_data()
                pass
            else:
                do_nothing()

        operations += 1


def run():
    global num_loops

    for _ in range(num_loops):
        main_loop()


def main():
    global operations, deletes, updates

    if len(sys.argv) < 6 or len(sys.argv) > 6:
        print("python3 tester.py <num_loops> <num_interactions> <pct_gdpr_users> <pct_delete> <pct_update>")
        exit(1)
    run()

    print("Total Operations: " + str(operations))
    print("Total Deletions:  " + str(deletes))
    print("Total Updates:    " + str(updates))

if __name__ == "__main__":
    main()
