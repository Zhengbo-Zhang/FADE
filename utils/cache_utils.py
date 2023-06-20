# Detections FIFOCache

from collections import deque


class DetCache(object):
    def __init__(self, capacity=10):
        self.capacity = capacity
        self._init(capacity)

    def size(self):
        """Return the approximate size of the cache."""
        return self._size()

    def empty(self):
        """Return True if the cache is empty, False otherwise."""
        return not self._size()

    def full(self):
        """Return True if the cache is full, False otherwise."""
        return 0 < self.capacity <= self._size()

    def put(self, det):
        """Put detections into the cache."""
        # TODO: put detections into cache
        pass

    def get(self):
        """Get aggregated result of detections."""
        # TODO: return overall result
        pass

    def _init(self, capacity):
        self.cache = deque(maxlen=capacity)

    def _size(self):
        return len(self.cache)

    def _put(self, item):
        self.cache.append(item)

    def _get(self):
        return self.cache.popleft()
