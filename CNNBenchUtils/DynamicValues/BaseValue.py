# TODO: implement Value Sets? e.g. v.value(stagenum, setnum)


class BaseValue:
    '''
    Base class for Dynamic Value types which are used to progressively change parameters in a multi-stage run of functions etc.
    If you should want to create a new derived class, you need to implement the value(), shortname() and parse() functions
    '''
    def __init__(self, val=None, stages=0, gapless=False):
        '''
        initialize the dynamic value.
        :param val: current value if any
        :param stages: over how many stages this will be used. Used to calculate progression when returning a new value
        :param gapless: changes the behaviour to avoid sudden value jumps, if the dynamic value hasn't been updated for a couple of stages
        '''
        # calculate new value without the gap caused by a lock
        self.gapless = gapless
        self.stages = int(stages)
        self.current_stage = 0
        # next stage after being unlocked again
        self.next_stage = 1
        self.locked = True
        self.val = val

    def __str__(self):
        return str(self.val)

    def __repr__(self):
        return str(self.val)

    @staticmethod
    def parse(param, stages, gapless):
        '''
        IMPLEMENT THIS!

        base function to be called by the actual implementation

        This function is supposed to parse a parameter section of a description file
        :param param: parameter section as a dict
        :param stages: see init
        :param gapless: see init
        :return: None type
        '''
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
        '''
        IMPLEMENT THIS!

        just returns its own shortened name as a string
        :return: short name of the instanced class
        '''
        return 'none'

    def value(self, stage=0):
        '''
        IMPLEMENT THIS!

        should return the current value according to the stage
        This is just a base function to be called by the actual implementation
        :param stage: stage for which the value should be calculated
        :return: None type
        '''
        if not self.is_locked() and self.gapless:
            if stage != self.current_stage:
                self.current_stage = stage
                self.next_stage += 1

        return None

    def actual_stage(self, stage):
        '''
        This is for "gapless"-mode and calculates a stage number to avoid value jumps
        :param stage:
        :return:
        '''
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