class BaseBackendRunner:
    def __init__(self, model, model_cfg, args, device):
        self.model = model
        self.model_cfg = model_cfg
        self.args = args
        self.device = device

    def infer(self, inputs):
        raise NotImplementedError
