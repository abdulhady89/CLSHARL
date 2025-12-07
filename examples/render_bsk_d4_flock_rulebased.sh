
# ARGS

model_path="/newEra3/maxsum/rf/CLSHARL/examples/"
# env_key="het_cloud_cluster-easy"
env_key="hmg_flock-easy"
algo="rulebased"



python render_bsk.py --algo ${algo} --env bsk --exp_name render --seed 0 --model_path ${model_path} --env_key ${env_key} 