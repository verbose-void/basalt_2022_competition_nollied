from fgz_config import FGZConfig

from vpt.run_agent import load_agent


def get_agent(config: FGZConfig, device=None):
    print("Loading model", config.model_filename)
    print("with weights", config.weights_filename)
    return load_agent(config.model_path, config.weights_path, device=device)
