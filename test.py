import torch
from src.utils import read_yaml_config_file
from src.main_utils import instantiate_lit_model


config = read_yaml_config_file("configs_runs/gans/stylegan_ore_maps_cond_w.yaml")
lit_model = instantiate_lit_model(config)
try:
    lit_model = lit_model.load_from_checkpoint(
        "logs/stylegan_ore_maps_cond_w/version_0/checkpoints/epoch=49-step=35200.ckpt"
    ).to(torch.device("cuda"))
except:
    lit_model = lit_model.load_from_checkpoint(
        "logs/stylegan_ore_maps_cond_w/version_0/checkpoints/epoch=49-step=35200.ckpt",
        **lit_model.hparams
    ).to(torch.device("cuda"))
input = torch.randn(1, 2, 32, 32).to(torch.device("cuda"))

lit_model.inference(input)
print("done")
