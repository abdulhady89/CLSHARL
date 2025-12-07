
# ARGS
# model_path="/newEra3/maxsum/project/CLSHARL/examples/results/bsk/het_cloud_cluster-easy-random_res_n_target/happo/2000tgt/seed-00000-2025-11-30-08-37-35/models"
model_path="/newEra3/maxsum/rf/CLSHARL/examples/results/bsk/hmg_flock-easy-random_res_n_target/mappo/flock/seed-00000-2025-11-26-17-20-46/models"
# env_key="het_cloud_cluster-easy"
env_key="hmg_flock-easy"
algo="mappo"



python render_bsk.py --algo ${algo} --env bsk --exp_name render --seed 0 --model_path ${model_path} --env_key ${env_key} 