algo="happo"
env="bsk"
seed_max=3

for seed in `seq ${seed_max}`;
do
    echo "Running ${algo} with ${env} environment and seed is ${seed}:"
    nohup python train.py --algo ${algo} --env ${env} --exp_name 'het_easy' --seed ${seed} > log_het_easy_${algo}${seed}.txt &
done