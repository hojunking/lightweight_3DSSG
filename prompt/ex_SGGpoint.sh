#!/bin/bash
run_command() {
    python -m main --mode $1 --exp $2 --part $3 --config $4 --st_ratio $5 --unst_ratio $6 --pretrained $7
}

#run_command train SGGpoint_test2 gcn ./config/SGGpoint.json 0 0 x
#run_command eval eval_test gcn ./config/SGGpoint.json 0 0 /home/song/Desktop/song/lightweight_3DSSG/config/ckp/SGGpoint/SGGpoint_test
run_command prune sggpoint_st25_gcn_sggpoint gcn ./config/SGGpoint.json 0.25 0 x
run_command prune sggpoint_st50_gcn_sggpoint gcn ./config/SGGpoint.json 0.50 0 x
run_command prune sggpoint_st75_gcn_sggpoint gcn ./config/SGGpoint.json 0.75 0 x

run_command prune sggpoint_unst25_gcn_sggpoint gcn ./config/SGGpoint.json 0 0.25 x
run_command prune sggpoint_unst50_gcn_sggpoint gcn ./config/SGGpoint.json 0 0.50 x
run_command prune sggpoint_unst75_gcn_sggpoint gcn ./config/SGGpoint.json 0 0.75 x