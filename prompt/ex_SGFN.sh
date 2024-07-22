#!/bin/bash
run_command() {
    python -m main --mode $1 --exp $2 --part $3 --config $4 --st_ratio $5 --unst_ratio $6 --pretrained $7
}

# Call the function with different sets of arguments

#run_command train real_baseline_param50_gcn_sgfn gcn ./config/SGFN.json 0 0

# load param50 baseline
# run_command prune param50_st25_gcn_sgfn gcn ./config/SGFN.json 0.25 0 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/real_baseline_param50_gcn_sgfn
# # load param50 baseline
# run_command prune param50_unst75_gcn_sgfn gcn ./config/SGFN.json 0 0.75 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/real_baseline_param50_gcn_sgfn

# # load st25
# run_command prune param50_st25_unst75_gcn_sgfn gcn ./config/SGFN.json 0 0.75 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/param50_st25_gcn_sgfn
# # load param50 baseline
# run_command prune param50_baseline_st25_unst75_gcn_sgfn gcn ./config/SGFN.json 0.25 0.75 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/real_baseline_param50_gcn_sgfn

# run_command eval eval_param50_st25_gcn_attn_sgfn gcn ./config/attn_SGFN.json 0.25 0 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/attn_sgfn_param50_gcn_sgfn
# run_command eval eval_param50_st50_gcn_attn_sgfn gcn ./config/attn_SGFN.json 0.5 0 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/attn_sgfn_param50_gcn_sgfn
# run_command eval eval_param50_st75_gcn_attn_sgfn gcn ./config/attn_SGFN.json 0.75 0 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/attn_sgfn_param50_gcn_sgfn

run_command eval eval_param50_unst25_gcn_attn_sgfn gcn ./config/attn_SGFN.json 0 0.25 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/attn_sgfn_param50_gcn_sgfn
run_command eval eval_param50_unst50_gcn_attn_sgfn gcn ./config/attn_SGFN.json 0 0.5 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/attn_sgfn_param50_gcn_sgfn
run_command eval eval_param50_unst75_gcn_attn_sgfn gcn ./config/attn_SGFN.json 0 0.75 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/attn_sgfn_param50_gcn_sgfn
