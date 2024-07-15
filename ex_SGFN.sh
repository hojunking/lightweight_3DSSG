#!/bin/bash
run_command() {
    python -m main --mode $1 --exp $2 --part $3 --config $4 --st_ratio $5 --unst_ratio $6 --pretrained $7
}

# Call the function with different sets of arguments


run_command prune st50_unst75_gcn_attn_sgfn gcn ./config/SGFN.json 0.5 0.75 /home/knuvi/Desktop/song/VLSAT_pruning/config/ckp/SGFN/SGFN_baseline
# run_command prune real_st25_gcn_sgfn gcn sgfn ./config/SGFN.json 0.25 st /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/real_sgfn_baseline
# run_command prune real_st50_gcn_sgfn gcn sgfn ./config/SGFN.json 0.5 st /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/real_sgfn_baseline
# run_command prune real_st75_gcn_sgfn gcn sgfn ./config/SGFN.json 0.75 st /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/real_sgfn_baseline

#run_command train param_reduction75_gcn_sgfn gcn sgfn ./config/mmgnet.json 0 st
#run_command train param_reduction25_gcn_sgfn gcn sgfn ./config/mmgnet.json 0 st
# run_command prune param25_unst75_gcn_sgfn gcn sgfn ./config/mmgnet.json 0.75 unst /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/param_reduction25_gcn_sgfn
# run_command prune param50_unst75_gcn_sgfn gcn sgfn ./config/mmgnet.json 0.75 unst /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/param_reduction50_gcn_sgfn
# run_command prune param75_unst75_gcn_sgfn gcn sgfn ./config/mmgnet.json 0.75 unst /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/param_reduction75_gcn_sgfn

# run_command prune param25_unst75_st50_gcn_sgfn gcn sgfn ./config/mmgnet.json 0.5 st /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/param25_unst75_gcn_sgfn
# run_command prune param50_unst75_st50_gcn_sgfn gcn sgfn ./config/mmgnet.json 0.5 st /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/param50_unst75_gcn_sgfn
# run_command prune param75_unst75_st50_gcn_sgfn gcn sgfn ./config/mmgnet.json 0.5 st /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGFN/param75_unst75_gcn_sgfn

#run_command prune pre_st75_gcn_sgfn gcn sgfn ./config/mmgnet.json 0.75 st
