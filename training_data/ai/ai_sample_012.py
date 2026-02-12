10.# Priority Queue using heapq
import heapq

class PriorityQueue:
    def __init__(self):
        self.pq = []

    def push(self, item, priority):
        heapq.heappush(self.pq, (priority, item))

    def pop(self):
        if self.pq:
            return heapq.heappop(self.pq)[1]
        return None