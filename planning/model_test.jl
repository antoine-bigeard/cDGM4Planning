using PyCall

py"""
import sys
import torch
sys.path.insert(0, "/Users/anthonycorso/Workspace/DeepGenerativeModelsCCS")
from src.utils import read_yaml_config_file
from src.main_utils import instantiate_lit_model
config = read_yaml_config_file("models/ddpm_ore_maps_500.yaml")
lit_model = instantiate_lit_model(config)
lit_model = lit_model.load_from_checkpoint("models/best-checkpoint.ckpt").to(torch.device('cpu'))
"""


model = py"lit_model"
input = py"torch.randn(1,2,32,32)"

model.inference(input)