import threading.Lock as Lock

class PCQueue:
    def __init__(self):
        q = []
        l = Lock()

    def deque(self):
        l.acquire()
        if len(q) > 0:
            return l.pop(0)
        else:
            return None
        l.release()

    def enque(self, item):
        l.acquire()
        q.append(item)
        l.release()
