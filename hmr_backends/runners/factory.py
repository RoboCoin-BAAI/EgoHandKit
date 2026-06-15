from .batch_runner import BatchBackendRunner
from .hawor_runner import HaworBackendRunner


def build_runner(bundle, args, device):
    if bundle.backend_name == 'hawor':
        return HaworBackendRunner(bundle.model, bundle.model_cfg, args, device)
    return BatchBackendRunner(bundle.model, bundle.model_cfg, args, device, bundle.backend_name)
