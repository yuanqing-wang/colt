from ..acquisition import Acquisition



class BayesianAcquisition(Acquisition):
    def __init__(self, model):
        super.__init__()
        self.model = model
        
    def train(self, past):
        self.model.train(past)