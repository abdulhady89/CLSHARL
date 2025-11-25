# model_path="/home/hady/project/CLSHARL/results/result_collection/homogeneous/easy/mappo/seed-00000-2025-10-17-17-46-18/models"
# env_key="hmg_cluster-easy"
# algo="mappo"

# EASY
# model_path="/home/hady/project/CLSHARL/examples/results/bsk/het_cluster-easy/mappo/het_easy/seed-00001-2025-10-20-03-53-30/models"
# env_key="het_cluster-easy"
# algo="mappo"

# # EASY
# model_path="/home/hady/project/CLSHARL/examples/results/bsk/het_cluster-easy/hatrpo/het_cluster-easy/seed-00001-2025-10-21-17-37-40/models"
# env_key="het_cluster-easy"
# algo="hatrpo"

# # EASY
# model_path="/home/hady/project/CLSHARL/examples/results/bsk/het_cluster-easy/happo/het_easy/seed-00001-2025-10-20-11-18-26/models"
# env_key="het_cluster-easy"
# algo="happo"


# EASY-RANDOM
# model_path="/home/hady/project/CLSHARL/examples/results/bsk/het_cluster-easy-random_res/happo/het_cluster-easy-random_all/seed-00000-2025-10-22-07-00-38/models"
# env_key="het_cluster-easy-random_res"
# algo="happo"

# EASY-RANDOM
# model_path="/home/hady/project/CLSHARL/examples/results/bsk/het_cluster-easy-random_res/mappo/het_cluster-easy-random_all/seed-00000-2025-10-22-07-00-03/models"
# env_key="het_cluster-easy-random_res"
# algo="mappo"

#EASY-RANDOM
# model_path="/home/hady/project/CLSHARL/examples/results/bsk/het_cluster-easy-random_all/hatrpo/het_cluster-easy-random_all/seed-00001-2025-10-21-06-05-38/models"
# env_key="het_cluster-easy-random_res"
# algo="hatrpo"

# HARD-RANDOM
# model_path="/home/hady/project/CLSHARL/examples/results/bsk/het_cluster-hard-random_all/happo/het_cluster-hard-random_all/seed-00001-2025-10-21-02-49-32/models"
# env_key="het_cluster-hard-random_res"
# algo="happo"

# HARD-RANDOM
# model_path="/home/hady/project/CLSHARL/examples/results/bsk/het_cluster-hard-random_all/mappo/het_cluster-hard-random_all/seed-00001-2025-10-20-22-25-51/models"
# env_key="het_cluster-hard-random_res"
# algo="mappo"

# HARD-RANDOM
model_path="/home/hady/project/CLSHARL/examples/results/bsk/het_cluster-hard-random_res/hatrpo/het_cluster-hard-random_all/seed-00001-2025-10-21-17-34-39/models"
env_key="het_cluster-hard-random_res"
algo="hatrpo"

# HARD
# model_path="/home/hady/project/CLSHARL/examples/results/bsk/het_cluster-hard/hatrpo/hard/seed-00001-2025-10-21-12-59-33/models"
# env_key="het_cluster-hard"
# algo="hatrpo"

# HARD
# model_path="/home/hady/project/CLSHARL/examples/results/bsk/het_cluster-hard/happo/het_cluster-hard/seed-00001-2025-10-20-17-39-53/models"
# env_key="het_cluster-hard"
# algo="happo"

# HARD
# model_path="/home/hady/project/CLSHARL/examples/results/bsk/het_cluster-hard/mappo/het_cluster-hard/seed-00001-2025-10-20-21-05-49/models"
# env_key="het_cluster-hard"
# algo="mappo"



python render_bsk.py --algo ${algo} --env bsk --exp_name render --seed 1 --model_path ${model_path} --env_key ${env_key}