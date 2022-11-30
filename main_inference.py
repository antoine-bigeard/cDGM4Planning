from src.utils import read_yaml_config_file
from src.main_utils import instantiate_lit_model


def inference(path_config, path_ckpt, conditions):
    config = read_yaml_config_file(path_config)
    lit_model = instantiate_lit_model(config)
    lit_model.load_from_checkpoint(path_ckpt)

    return lit_model.inference(conditions)
