# Detections FIFOCache

import torch


class DetCache(object):
    def __init__(self, capacity=10):
        self.capacity = capacity
        self.queue = list()

    def size(self):
        """Return the approximate size of the cache."""
        return self._qsize()

    def empty(self):
        """Return True if the cache is empty, False otherwise."""
        return not self._qsize()

    def full(self):
        """Return True if the cache is full, False otherwise."""
        return 0 < self.capacity <= self._qsize()

    def put(self, det):
        """Put detections into the cache."""
        if self.full():
            self._get()
        self._put(det)

    def get(self):
        """Get aggregated result of detections."""
        # TODO: return overall result
        return max(self.queue) > 0.7 or (self.full() and all(torch.tensor(self.queue) > 0.3))

    def _qsize(self):
        return len(self.queue)

    def _put(self, item):
        self.queue.append(item)

    def _get(self):
        return self.queue.pop(0)
