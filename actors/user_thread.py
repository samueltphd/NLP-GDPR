import random
import time

def user_thread(obj, agg, log, train_q, weight_q, stop_q, data_reserves, delete_pct, update_pct):
    start = time.time()
    print("Starting user thread at: " + str(start))

    while True:
        if stop_q.deque() is not None:
            break

        # receive training requests from server thread
        trequest = train_q.deque()

        if trequest is not None:
            print("[user thread " + str(obj.uid) + "] starting to train")

            try:
                # call the training function with the appropriate global weights
                # and the appropriate round information
                result = trequest[0](trequest[1], trequest[2])
                print("[user thread " + str(obj.uid) + "] training complete")

            except ValueError:
                result = trequest[2]
                print("[user thread " + str(obj.uid) + "] training error... passing back prior global weights")

            # send the results back to the aggregator
            weight_q.enque(result)

            # determine any requests to be made
            # if so send request to aggregator to update logger
            count = 0
            for _ in range(100):
                r = random.randrange(100)
                i = random.randrange(len(obj.data))

                if r < delete_pct:
                    obj.change_data_permission(i)
                    obj.request_aggregator_update(agg)
                    count += 1
                elif r < delete_pct + update_pct:
                    to_update = obj.data_id_to_data_point[i]
                    to_update['val'] = data_reserves.sample()['body'].values[0]
                    obj.update_data(i, to_update)
                    count += 1
                else:
                    d = data_reserves.sample()
                    obj.add_data({'id': len(obj.data), 'val': d['body'].values[0], 'target': d['target'].values[0]})

            print("[user thread " + str(obj.uid) + "] requested the aggregator to update " + str(count) + " times...")

            time.sleep(5)

    print("[user thread " + str(obj.uid) + "] exiting...")
