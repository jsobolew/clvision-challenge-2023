#!/bin/bash

si_lambdas=(0.2 0.5 0.6 0.7 0.8)

for i in 1 2 3 4 5
do
   echo "running: /home/jsobolewski/anaconda3/envs/clvision23-challenge/bin/python /home/jsobolewski/clvision-challenge-2023/train.py --run_name SI_full_${si_lambdas[$i]} --si_lambda ${si_lambdas[$i]} --config_file config_s3.pkl"
   /home/jsobolewski/anaconda3/envs/clvision23-challenge/bin/python /home/jsobolewski/clvision-challenge-2023/train.py --run_name SI_full_${si_lambdas[$i]} --si_lambda ${si_lambdas[$i]} --config_file config_s3.pkl
done

