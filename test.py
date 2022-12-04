import torch
from src.utils import read_yaml_config_file
from src.main_utils import instantiate_lit_model


config = read_yaml_config_file("configs_runs/ddpm_ore_maps_500.yaml")
lit_model = instantiate_lit_model(config)
lit_model = lit_model.load_from_checkpoint(
    "logs/ore_maps_ddpm_250/version_0/checkpoints/best-checkpoint.ckpt"
).to(torch.device("cuda"))

input = torch.randn(1, 2, 32, 32).to(torch.device("cuda"))

lit_model.inference(input)
