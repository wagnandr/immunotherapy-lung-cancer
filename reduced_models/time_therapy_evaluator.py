import numpy as np 

class TimeTherapyEvaluator:
    def __init__(self, start, length):
        self.start_therapy = start
        self.length_therapy = length

        self._start_application = np.array(self.start_therapy)-1e-15
        self._end_application = np.array(self.start_therapy)+1e-15 + np.array(self.length_therapy)
    
    def __call__(self, t):
        s = self._start_application
        e = self._end_application
        feasible = np.any((np.repeat(s, len(t)).reshape(-1, len(t)) <= t) * (t <= np.repeat(e, len(t)).reshape(-1, len(t))), axis=0)
        return feasible 

        '''
        for start, length in zip(self.start_therapy, self.length_therapy):
            if t >= start-1e-15 and t <= start+length+1e-15:
                return 1.0 
        return 0. 
        '''