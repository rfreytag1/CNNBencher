# TODO: implement Value Sets? e.g. v.value(stagenum, setnum)


class BaseValue:
    def __init__(self, val=None, stages=0, gapless=False):
        self.gapless = gapless  # calculate new value without the gap caused by a lock
        self.stages = int(stages)
        self.current_stage = 0
        self.next_stage = 1  # next stage after being unlocked again
        self.locked = True
        self.val = val

    def __str__(self):
        return str(self.val)

    def __repr__(self):
        return str(self.val)

    @staticmethod
    def parse(param, stages, gapless):
        if not isinstance(param, dict):
            raise TypeError('Parameter "param" must be a dict!')
        if not isinstance(stages, int):
            raise TypeError('Parameter "stages" must be an integer!')
        if not isinstance(gapless, bool):
            raise TypeError('Parameter "gapless" must be a boolean!')

        if 'type' not in param:
            raise ValueError('Dict "param" is missing field "type"!')

        return None

    @staticmethod
    def shortname():
        return 'none'

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