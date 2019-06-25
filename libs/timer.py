import time

class Timer():
    def __init__(self):
        self.diff_n = 0
        self.startPoint_n = 0
        self.stack = []
    
    def start(self):
        self.startPoint_n = time.time()
        return self

    def stop(self):
        self.diff_n = time.time() - self.startPoint_n
        return self

    def getPassedTime(self):
        return int(self.diff_n)

    def note(self, label):
        self.stack.append((label, self.getPassedTime()))
        return self