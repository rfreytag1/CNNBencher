# TODO: implement Value Sets? e.g. v.value(stagenum, setnum)


class BaseValue:
    def __init__(self, val=None, stages=0):
        self.stages = stages
        self.locked = True
        self.val = val

    def __str__(self):
        return str(self.val)

    def __repr__(self):
        return str(self.val)

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