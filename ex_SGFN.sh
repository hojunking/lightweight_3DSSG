#!/bin/bash
run_command() {
    python -m main --mode $1 --exp $2 --part $3 --model $4 --config $5 --ratio $6 --method $7 --pretrained $8
}

# Call the function with different sets of arguments


run_command train param50_real_real_sgfn_baseline gcn sgfn ./config/SGFN.json 0 unst x
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
