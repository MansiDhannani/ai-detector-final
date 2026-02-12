3# Queue Class
class QueueNode:
    def __init__(self, value):
        self.value = value
        self.next = None

class Queue:
    def __init__(self):
        self.front = None
        self.rear = None

    def enqueue(self, value):
        new_node = QueueNode(value)
        if self.rear:
            self.rear.next = new_node
        self.rear = new_node
        if self.front is None:
            self.front = new_node

    def dequeue(self):
        if self.front is None:
            return None
        dequeued_value = self.front.value
        self.front = self.front.next
        if self.front is None:
            self.rear = None
        return dequeued_value

    def peek(self):
        return self.front.value if self.front else None

    def is_empty(self):
        return self.front is None