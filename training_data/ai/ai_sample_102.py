100.# Simple State Machine
class StateMachine:
    def __init__(self, state):
        self.state = state

    def transition(self, new_state):
        print(f"{self.state} -> {new_state}")
        self.state = new_state