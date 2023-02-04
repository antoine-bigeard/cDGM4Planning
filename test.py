import torch
from src.utils import read_yaml_config_file
from src.main_utils import instantiate_lit_model


config = read_yaml_config_file("logs/ddpms/ddpm_ore_maps_250_medium/version_0/ddpm_ore_maps_250_medium.yaml")
lit_model = instantiate_lit_model(config)
try:
    lit_model = lit_model.load_from_checkpoint("logs/ddpms/ddpm_ore_maps_250_medium/version_0/checkpoints/ddpm_ore_maps_250_medium.ckpt").to(torch.device("cuda"))
except:
    lit_model = lit_model.load_from_checkpoint("logs/ddpms/ddpm_ore_maps_250_medium/version_0/checkpoints/ddpm_ore_maps_250_medium.ckpt", **lit_model.hparams).to(torch.device("cuda"))
input = torch.randn(1, 2, 32, 32).to(torch.device("cuda"))

lit_model.inference(input)