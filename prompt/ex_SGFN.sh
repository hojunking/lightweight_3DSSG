#!/bin/bash
run_command() {
    python -m main --mode $1 --exp $2 --part $3 --config $4 --st_ratio $5 --unst_ratio $6 --pretrained $7
}
# Test
#run_command eval test gcn ./config/SGFN.json 0.80 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/real_baseline_sgfn
#run_command eval testtest gcn ./config/SGFN.json 0 0.5 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/real_baseline_sgfn
run_command prune test gnn ./config/attn_SGFN.json 0.25 0 x

# SGFN
#run_command prune sgfn_baseline_st35_gcn_sgfn gnn ./config/SGFN.json 0.35 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/real_baseline_sgfn
# run_command eval eval_redu60_st10_gcn_sgfn gcn ./config/SGFN.json 0.10 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/redu60_epoch110_gcn_sgfn
# run_command eval eval_redu60_st15_gcn_sgfn gcn ./config/SGFN.json 0.15 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/redu60_epoch110_gcn_sgfn
# run_command eval eval_redu60_st20_gcn_sgfn gcn ./config/SGFN.json 0.20 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/redu60_epoch110_gcn_sgfn
# run_command eval eval_redu60_st25_gcn_sgfn gcn ./config/SGFN.json 0.25 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/redu60_epoch110_gcn_sgfn
# run_command eval eval_redu60_st30_gcn_sgfn gcn ./config/SGFN.json 0.30 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/redu60_epoch110_gcn_sgfn
# run_command eval eval_redu60_st35_gcn_sgfn gcn ./config/SGFN.json 0.35 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/redu60_epoch110_gcn_sgfn
# run_command eval eval_redu60_st40_gcn_sgfn gcn ./config/SGFN.json 0.40 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/redu60_epoch110_gcn_sgfn
# run_command eval eval_redu60_st45_gcn_sgfn gcn ./config/SGFN.json 0.45 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/redu60_epoch110_gcn_sgfn
# run_command eval eval_redu60_st50_gcn_sgfn gcn ./config/SGFN.json 0.50 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/redu60_epoch110_gcn_sgfn
# run_command eval eval_redu60_st55_gcn_sgfn gcn ./config/SGFN.json 0.55 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/redu60_epoch110_gcn_sgfn
# run_command eval eval_redu60_st60_gcn_sgfn gcn ./config/SGFN.json 0.60 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/redu60_epoch110_gcn_sgfn
# run_command eval eval_redu60_st65_gcn_sgfn gcn ./config/SGFN.json 0.65 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/redu60_epoch110_gcn_sgfn
# run_command eval eval_redu60_st70_gcn_sgfn gcn ./config/SGFN.json 0.70 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/redu60_epoch110_gcn_sgfn
# run_command eval eval_redu60_st75_gcn_sgfn gcn ./config/SGFN.json 0.75 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/redu60_epoch110_gcn_sgfn

# run_command eval eval_sgfn_unst80_gcn_sgfn gcn ./config/SGFN.json 0 0.80 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/real_baseline_sgfn
# run_command eval eval_sgfn_unst85_gcn_sgfn gcn ./config/SGFN.json 0 0.85 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/real_baseline_sgfn
# run_command eval eval_sgfn_unst90_gcn_sgfn gcn ./config/SGFN.json 0 0.90 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/real_baseline_sgfn
# run_command eval eval_sgfn_unst95_gcn_sgfn gcn ./config/SGFN.json 0 0.95 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/real_baseline_sgfn
#run_command train redu60_attn_dim128_gcn_sgfn gcn ./config/SGFN.json 0 0 x
#run_command prune redu60_attn_dim128_st25_gcn_sgfn gcn ./config/SGFN.json 0.25 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/redu60_attn_dim128_gcn_sgfn

# run_command prune redu50_st25_sgfn gcn ./config/SGFN2.json 0.25 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/real_baseline_param50_gcn_sgfn
# #run_command prune redu60_attn_dim128_st25_gcn_sgfn gcn ./config/SGFN.json 0.25 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/redu60_attn_dim128_gcn_sgfn
# run_command train redu70_attn_edge_dim128_gcn_sgfn gcn ./config/SGFN3.json 0 0 x
# run_command prune redu70_attn_edge_dim128_st25_gcn_sgfn gcn ./config/SGFN3.json 0.25 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/redu70_attn_edge_dim128_gcn_sgfn

# # attn+SGFN
#run_command prune attnSGFN_baseline_st20_gcn_attn_sgfn gnn ./config/attn_SGFN.json 0.20 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/selfattn_SGFN_baseline
# run_command train redu55_edge_dim128_gcn_attn_sgfn gcn ./config/attn_SGFN.json 0 0 x
# run_command prune redu55_edge_dim128_st25_gcn_attn_sgfn gcn ./config/attn_SGFN.json 0.25 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/redu55_edge_dim128_gcn_attn_sgfn
# run_command prune redu60_attn_edge_dim128_st25_gcn_attn_sgfn gcn ./config/attn_SGFN2.json 0.25 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/redu60_attn_edge_dim128_gcn_attn_sgfn

# run_command eval eval_redu60_st05_gcn_attnSGFN gcn ./config/attn_SGFN.json 0.05 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/redu60_epoch120_gcn_attn_sgfn
# run_command eval eval_redu60_st10_gcn_attnSGFN gcn ./config/attn_SGFN.json 0.10 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/redu60_epoch120_gcn_attn_sgfn
# run_command eval eval_redu60_st15_gcn_attnSGFN gcn ./config/attn_SGFN.json 0.15 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/redu60_epoch120_gcn_attn_sgfn
# run_command eval eval_redu60_st20_gcn_attnSGFN gcn ./config/attn_SGFN.json 0.20 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/redu60_epoch120_gcn_attn_sgfn
# run_command eval eval_redu60_st25_gcn_attnSGFN gcn ./config/attn_SGFN.json 0.25 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/redu60_epoch120_gcn_attn_sgfn
# run_command eval eval_redu60_st30_gcn_attnSGFN gcn ./config/attn_SGFN.json 0.30 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/redu60_epoch120_gcn_attn_sgfn
# run_command eval eval_redu60_st35_gcn_attnSGFN gcn ./config/attn_SGFN.json 0.35 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/redu60_epoch120_gcn_attn_sgfn
# run_command eval eval_redu60_st40_gcn_attnSGFN gcn ./config/attn_SGFN.json 0.40 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/redu60_epoch120_gcn_attn_sgfn
# run_command eval eval_redu60_st45_gcn_attnSGFN gcn ./config/attn_SGFN.json 0.45 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/redu60_epoch120_gcn_attn_sgfn
# run_command eval eval_redu60_st50_gcn_attnSGFN gcn ./config/attn_SGFN.json 0.50 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/redu60_epoch120_gcn_attn_sgfn
# run_command eval eval_redu60_st55_gcn_attnSGFN gcn ./config/attn_SGFN.json 0.55 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/redu60_epoch120_gcn_attn_sgfn
# run_command eval eval_redu60_st60_gcn_attnSGFN gcn ./config/attn_SGFN.json 0.60 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/redu60_epoch120_gcn_attn_sgfn
# run_command eval eval_redu60_st65_gcn_attnSGFN gcn ./config/attn_SGFN.json 0.65 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/redu60_epoch120_gcn_attn_sgfn
# run_command eval eval_redu60_st70_gcn_attnSGFN gcn ./config/attn_SGFN.json 0.70 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/redu60_epoch120_gcn_attn_sgfn
# run_command eval eval_redu60_st75_gcn_attnSGFN gcn ./config/attn_SGFN.json 0.75 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/redu60_epoch120_gcn_attn_sgfn

# run_command eval eval_attnSGFN_unst80_gcn_attnSGFN gcn ./config/attn_SGFN.json 0 0.80 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/selfattn_SGFN_baseline
# run_command eval eval_attnSGFN_unst85_gcn_attnSGFN gcn ./config/attn_SGFN.json 0 0.85 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/selfattn_SGFN_baseline
# run_command eval eval_attnSGFN_unst90_gcn_attnSGFN gcn ./config/attn_SGFN.json 0 0.90 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/selfattn_SGFN_baseline
# run_command eval eval_attnSGFN_unst95_gcn_attnSGFN gcn ./config/attn_SGFN.json 0 0.95 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/selfattn_SGFN_baseline
# run_command eval eval_attnSGFN_st70_gcn_attnSGFN gcn ./config/attn_SGFN.json 0.70 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/selfattn_SGFN_baseline
# run_command eval eval_attnSGFN_st75_gcn_attnSGFN gcn ./config/attn_SGFN.json 0.75 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/selfattn_SGFN_baseline
