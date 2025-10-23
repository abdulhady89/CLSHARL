algo="haa2c"
env="bsk"
seed_max=3
exp_name='het_easy'

for seed in `seq ${seed_max}`;
do
    echo "Running ${algo} with ${env} environment and seed is ${seed}:"
    nohup python train.py --algo ${algo} --env ${env} --exp_name ${exp_name} --seed ${seed} > log_${exp_name}_${algo}${seed}.txt &
done