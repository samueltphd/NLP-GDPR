import random
import time

update_pct = 30
delete_pct = 30

def user_thread(obj, _tlength, train_q, weight_q, update_q, delete_q):
    global update_pct, delete_pct

    start = time.time()
    print("Starting server thread at: " + str(start))

    while time.time() - start < _tlength:
        # receive training requests from server thread

        trequest = train_q.deque()

        if trequest is not None:
            weight_q.enque(obj.train(trequest))

        # determine any requests to be made to the aggregator
        r = random.randint(1, 100)

        if r < update_pct:
            def update():
                pass

            update_q.enque(update)

        elif r < update_pct + delete_pct:
            def delete():
                pass

            delete_q.enque(delete)
