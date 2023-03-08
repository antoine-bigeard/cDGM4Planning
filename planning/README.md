# Planning Experiments for **Conditional Deep Generative Models for Belief State Planning**

## File Descriptions

* `minex_definition.jl` - This defines the Mineral Exploration POMDP used in the experiments
* `voi_policy.jl` - This defines the VOI policy described in the paper
* `generative_ME_belief.jl` - This contains the definition of the cDGM belief. It interfaces with python so pytorch models can be used
* `belief_metrics.jl` - This file evaluates belief representations on task-agnostic and task-specific metrics and outputs the results in tables and plots
* `run_experiment.jl` - This contains the code that processes config files from `configs` and kicks off the desired experiments
* `process_results.jl` - This file processes the output from the `run_experiment.jl` file to produce plots and figures
* `run_baselines.sh` - This bash script will run the planning baseline experiments
* `run_cDGMs.sh` - This bash script will run the planning experiments with the cDGMs

## Configurations
The config files to run the experiments (and many other examples) are stored in the `configs` folder

## Models and Data
Pre-trained models and data can be provided upon request.