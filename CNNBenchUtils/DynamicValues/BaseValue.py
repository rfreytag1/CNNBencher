# TODO: implement Value Sets? e.g. v.value(stagenum, setnum)


class BaseValue:
    def __init__(self, val=None, stages=0, gapless=False):
        self.gapless = gapless  # calculate new value without the gap caused by a lock
        self.stages = int(stages)
        self.current_stage = 0
        self.next_stage = 0  # next stage after being unlocked again
        self.locked = True
        self.val = val

    def __str__(self):
        return str(self.val)

    def __repr__(self):
        return str(self.val)

    def value(self, stage=0):
        if not self.is_locked() and self.gapless:
            if stage != self.current_stage:
                self.current_stage = stage
                self.next_stage += 1

        return None

    def actual_stage(self, stage):
        if self.gapless:
            return (self.next_stage - 1) if (self.next_stage - 1) >= 0 else 0
        else:
            return stage

    # locking mechanism to keep values static if needed
    def lock(self):
        self.locked = True

    def unlock(self):
        self.locked = False

    def toggle_lock(self):
        self.locked = ~self.locked

    def is_locked(self):
        return self.locked