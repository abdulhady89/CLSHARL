
# ARGS
model_path="/newEra/maxsum/project/CLSHARL/examples/results/bsk/het_cloud_cluster-easy-random_res_n_target/happo/2000tgt/seed-00000-2025-11-30-08-37-35/models"
env_key="het_cloud_cluster-easy"
algo="happo"



python render_bsk.py --algo ${algo} --env bsk --exp_name render --seed 0 --model_path ${model_path} --env_key ${env_key}