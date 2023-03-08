# Conditional Deep Generative Models for Belief State Planning

This GitHub repository contains the implementation for the paper *Conditional Deep Generative Models for Belief State Planning*.

## Installation

To install the requirements, you may use the `poetry.lock` (recommended) or the `requirements.txt` file. The data we used for our experiments are also available in the `data` folder.

To run trainings, tests or inferences, we use yaml configuration files.

### Training a model

1. Create an appropriate configuration file. All the config files that we used for our trainings are provided. For the classic GAN, you may vary the whole architecture, using blocks that are implemented in `src/model/model/blocks.py`. For the DDPMs, parameters for the diffusion as well as the size (small, medium, normal) are changeable. For the StyleGAN, the architecture can not be changed in the configuration files.

2. Run the command: `python src/main.py --path_config path_to_your_config_file`. The experiments' results, tensorboard, checkpoints will be saved in the folder `logs/name_experiment/version_0`, where *name_experiment* is specified the config file. *version_i* folder is here to make sure a new experiment with the same name does not overwrite the previous one. It contains:
- checkpoints saved during the experiment in the folder `logs/name_experiment/checkpoints`.
- configuration file used for the run in the file `logs/name_experiment/config.yaml`.
- hparams of the lighting model in the file `logs/name_experiment/hparams.yaml` (you should not need to use this file).
- tests made at the beginning of each epoch in the folders `logs/name_experiment/epoch_i`.
- tensorboard results in a tensorboard events file.

3. A typical configuration file description for DDPMs can be found [here](configs_runs/ddpm_ore_maps_100_example.yaml). And for the GANs [here](configs_runs/gan_ore_maps_example.yaml).

### Testing a model

#### Metric based tests
First, it is possible to test the models using metrics of your choice (L1, MSE and distance_to_condition available for now). This part shows how to test a model
using those quantitative measures.

Several options are possible to test a model:

1. Re-use a training config file and change the mode from *fit* to *test*. Then run `python src/main.py --path_config path_to_your_test_config_file`.

2. To automatically use the checkpoint and configuration from a previous experiment:
- Prepare an `update_config_for_tests.yaml` file where you must give the path to your experiment's logs `logs/name_experiment/version_0`.
By default, this will take the original configuration that was used for training. However, it is possible to overwrite parameters if needed (for instance to change the batch_size, 
or the number of samples used for the tests). An example if provided below, and configurations are available on this repo.
- Run the command `python src/full_metric_test.py --path_config update_config_for_tests.yaml`.

To visualize the results of the tests, configure a `visualize.yaml` file to basically specify which tests results to plot, and where to save them.
Then run the command `python src/visualize_tests.py --path_config visualize.yaml`.

An example for the update config file can be found [here](configs_tests/update_conf_for_tests.yaml).

An example for the visualization config file can be found [here](configs_visualize/visualize.yaml).

#### Inference and image-based tests
This part shows how to test models on various data, and produce the output image to visually have a grasp of the model's performance. To do so:

1. Configure [this file](configs_cond_tests/image_test.yaml). You basically need to provide a list of number of conditions for the test, as well as the models to test, and a path to save the results
(optionally a path for previous tests that you want to plot again).

2. Run the command `python src/inference_test.py --path_config image_test.yaml`.

## Planning
For information about the planning experiments, see the README in the folder named `planning`