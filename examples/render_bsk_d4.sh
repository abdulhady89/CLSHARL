
# ARGS

# model_path="/home/hady/project/CLSHARL/examples/pretrained/happo/happo_scan_au_hard_models"
model_path="/home/hady/project/CLSHARL/examples/pretrained/hatrpo/hatrpo_scan_au_hard_models"
# model_path="/home/hady/project/CLSHARL/examples/pretrained/mappo/mappo_scan_au_hard_models"
# model_path="/home"


env_key="het_cloud_cluster-hard"
# env_key="het_cloud_cluster-hard-random_res_tgt"


# algo="happo"
algo="hatrpo"
# algo="mappo"
# algo="rulebased"


echo "Rendering BSK environment under $env_key setting with $algo policy.."


python render_bsk.py --algo ${algo} --env bsk --exp_name render_scan_d4_logged --seed 0 --model_path ${model_path} --env_key ${env_key}