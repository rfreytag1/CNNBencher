class BaseValue:
    def __init__(self, stages=0):
        self.stages = stages
        self.locked = True

    def value(self, stage):
        raise NotImplementedError()

    # locking mechanism to keep values static if needed
    def lock(self):
        self.locked = True

    def unlock(self):
        self.locked = False

    def toggle_lock(self):
        self.locked = ~self.locked

    def is_locked(self):
        return self.locked