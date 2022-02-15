import sys
import threading
import time

class AutoFlushThread(threading.Thread):
    __isRun = False
    def __init__(self, freq:float):
        threading.Thread.__init__(self)
        self.setDaemon(True)
        self.__freq = freq
        self.setName("AutoFlushThread")
    def run(self):
        if self.__isRun:
            return
        self.__isRun = True
        while True:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdin.flush()
            time.sleep(self.__freq)
    @staticmethod
    def flush(freq: float):
        aft = AutoFlushThread(freq)
        aft.start()

    def __del__(self):
        print("Stop flush")