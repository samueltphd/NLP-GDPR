from threading import Lock

class PCQueue:
    def __init__(self):
        self.q = []
        self.l = Lock()

    def deque(self):
        self.l.acquire()
        if len(self.q) > 0:
            self.l.release()
            return self.l.pop(0)
        else:
            self.l.release()
            return None

    def enque(self, item):
        self.l.acquire()
        self.q.append(item)
        self.l.release()
