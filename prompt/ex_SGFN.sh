#!/bin/bash
run_command() {
    python -m main --mode $1 --exp $2 --part $3 --config $4 --st_ratio $5 --unst_ratio $6 --pretrained $7
}

# Call the function with different sets of arguments

#run_command train real_baseline_param50_gcn_sgfn gcn ./config/attn_SGFN.json 0 0

# load param50 baseline
# run_command prune param50_st25_gcn_sgfn gcn ./config/attn_SGFN.json 0.25 0 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/real_baseline_param50_gcn_sgfn
# # load param50 baseline
# run_command prune param50_unst75_gcn_sgfn gcn ./config/attn_SGFN.json 0 0.75 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/real_baseline_param50_gcn_sgfn

# # load st25
# run_command prune param50_st25_unst75_gcn_sgfn gcn ./config/SGFN.json 0 0.75 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/param50_st25_gcn_sgfn
# # load param50 baseline
# run_command prune param50_baseline_st25_unst75_gcn_sgfn gcn ./config/SGFN.json 0.25 0.75 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/real_baseline_param50_gcn_sgfn
#run_command eval test gcn ./config/attn_SGFN.json 0.05 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/selfattn_SGFN_baseline

run_command eval eval_attnSGFN_st05_gcn_attnSGFN gcn ./config/attn_SGFN.json 0.05 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/selfattn_SGFN_baseline
run_command eval eval_attnSGFN_st10_gcn_attnSGFN gcn ./config/attn_SGFN.json 0.1 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/selfattn_SGFN_baseline
run_command eval eval_attnSGFN_st15_gcn_attnSGFN gcn ./config/attn_SGFN.json 0.15 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/selfattn_SGFN_baseline
run_command eval eval_attnSGFN_st20_gcn_attnSGFN gcn ./config/attn_SGFN.json 0.20 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/selfattn_SGFN_baseline
run_command eval eval_attnSGFN_st25_gcn_attnSGFN gcn ./config/attn_SGFN.json 0.25 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/selfattn_SGFN_baseline
run_command eval eval_attnSGFN_st30_gcn_attnSGFN gcn ./config/attn_SGFN.json 0.30 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/selfattn_SGFN_baseline
run_command eval eval_attnSGFN_st35_gcn_attnSGFN gcn ./config/attn_SGFN.json 0.35 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/selfattn_SGFN_baseline
run_command eval eval_attnSGFN_st40_gcn_attnSGFN gcn ./config/attn_SGFN.json 0.40 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/selfattn_SGFN_baseline
run_command eval eval_attnSGFN_st45_gcn_attnSGFN gcn ./config/attn_SGFN.json 0.45 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/selfattn_SGFN_baseline
run_command eval eval_attnSGFN_st50_gcn_attnSGFN gcn ./config/attn_SGFN.json 0.50 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/selfattn_SGFN_baseline
run_command eval eval_attnSGFN_st55_gcn_attnSGFN gcn ./config/attn_SGFN.json 0.55 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/selfattn_SGFN_baseline
run_command eval eval_attnSGFN_st60_gcn_attnSGFN gcn ./config/attn_SGFN.json 0.60 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/selfattn_SGFN_baseline
run_command eval eval_attnSGFN_st65_gcn_attnSGFN gcn ./config/attn_SGFN.json 0.65 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/selfattn_SGFN_baseline
run_command eval eval_attnSGFN_st70_gcn_attnSGFN gcn ./config/attn_SGFN.json 0.70 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/selfattn_SGFN_baseline
run_command eval eval_attnSGFN_st75_gcn_attnSGFN gcn ./config/attn_SGFN.json 0.75 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/selfattn_SGFN_baseline
