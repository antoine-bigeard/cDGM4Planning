export CUDA_VISIBLE_DEVICES=0
# export CUDA_VISIBLE_DEVICES=1
julia1.8 planning/run_experiment.jl planning/configs/ddpm250_voi_multi.yaml
# julia1.8 planning/run_experiment.jl planning/configs/ddpm250med_voi_multi.yaml
julia1.8 planning/run_experiment.jl planning/configs/ddpm500_voi_multi.yaml
# julia1.8 planning/run_experiment.jl planning/configs/ddpm500med_voi_multi.yaml
# julia1.8 planning/run_experiment.jl planning/configs/gan_half_2_32_w.yaml
# julia1.8 planning/run_experiment.jl planning/configs/gan_all_2_128_w.yaml

# julia1.8 planning/run_experiment.jl planning/configs/ddpm250_tree_search.yaml



