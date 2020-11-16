import random
import time

update_pct = 30
delete_pct = 30

def user_thread(obj, _tlength, train_q, weight_q, update_q, delete_q, data_reserves):
    global update_pct, delete_pct

    start = time.time()
    print("Starting user thread at: " + str(start))

    while time.time() - start < _tlength:
        # receive training requests from server thread

        trequest = train_q.deque()

        if trequest is not None:
            weight_q.enque(obj.train(trequest))

        # determine any requests to be made to the aggregator
        r = random.randint(1, 100)

        if r < update_pct:
            new_data = data_reserves.sample()['body']
            if len(obj.data) > 0:
                to_update = random.choice(obj.data)

                obj.update_data(to_update['id'], {'id': to_update['id'],
                 'val': new_data,
                 'rids': to_update['rids'] if 'rids' in to_update.keys() else [],
                 'opt_in': to_update['opt_in']
                 })

        elif r < update_pct + delete_pct:
            if len(obj.data) > 0:
                to_remove = random.choice(obj.data)

                obj.remove_data(to_remove['id'])
