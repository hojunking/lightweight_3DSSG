#!/bin/bash
run_command() {
    python -m main --mode $1 --exp $2 --part $3 --config $4 --st_ratio $5 --unst_ratio $6 --pretrained $7
}
# Test
#run_command prune test gcn ./config/attn_SGFN.json 0.1 0 x

# SGFN 
#run_command train redu60_attn_dim128_gcn_sgfn gcn ./config/SGFN.json 0 0 x
#run_command prune redu60_attn_dim128_st25_gcn_sgfn gcn ./config/SGFN.json 0.25 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/redu60_attn_dim128_gcn_sgfn

run_command prune redu50_st25_sgfn gcn ./config/SGFN2.json 0.25 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/real_baseline_param50_gcn_sgfn
run_command prune redu60_attn_dim128_st25_gcn_sgfn gcn ./config/SGFN.json 0.25 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/redu60_attn_dim128_gcn_sgfn
run_command train redu70_attn_edge_dim128_gcn_sgfn gcn ./config/SGFN3.json 0 0 x
run_command prune redu70_attn_edge_dim128_st25_gcn_sgfn gcn ./config/SGFN3.json 0.25 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/redu70_attn_edge_dim128_gcn_sgfn

# attn+SGFN
run_command train redu55_edge_dim128_gcn_attn_sgfn gcn ./config/attn_SGFN.json 0 0 x
run_command prune redu55_edge_dim128_st25_gcn_attn_sgfn gcn ./config/attn_SGFN.json 0.25 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/redu55_edge_dim128_gcn_attn_sgfn
run_command prune redu60_attn_edge_dim128_st25_gcn_attn_sgfn gcn ./config/attn_SGFN2.json 0.25 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/redu60_attn_edge_dim128_gcn_attn_sgfn



# run_command eval eval_attnSGFN_st70_gcn_attnSGFN gcn ./config/attn_SGFN.json 0.70 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/selfattn_SGFN_baseline
# run_command eval eval_attnSGFN_st75_gcn_attnSGFN gcn ./config/attn_SGFN.json 0.75 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/selfattn_SGFN_baseline
