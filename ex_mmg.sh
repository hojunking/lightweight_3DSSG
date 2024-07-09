#!/bin/bash
run_command() {
    python -m main --mode $1 --exp $2 --part $3 --model $4 --config $5 --ratio $6 --method $7 --pretrained $8

}

# Call the function with different sets of arguments

# Run 1
#run_command prune st_enco_prune50 encoder sgfn ./config/mmgnet.json

# Run 2
#run_command prune st_gcn_prune50 gcn sgfn ./config/mmgnet.json

# Run 3
#run_command prune st_classif_prune50 classifier sgfn ./config/mmgnet.json

# Run 4
#run_command prune origin_t3 gcn Mmgnet ./config/mmgnet.json 0.3

#run_command train test gcn Mmgnet ./config/mmgnet.json 0 unst
run_command prune test gcn Mmgnet ./config/mmgnet.json 0.75 unst /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/p_redu25

#run_command prune param25_unst75_gcn_mmg gcn Mmgnet ./config/mmgnet.json 0.75 unst /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/p_redu25
#run_command prune param25_unst75_st50_gcn_mmg gcn Mmgnet ./config/mmgnet.json 0.5 st /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/param25_unst75_gcn_mmg

# run_command prune param50_unst75_gcn_mmg gcn Mmgnet ./config/mmgnet.json 0.75 unst /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/p_redu50
# run_command prune param50_unst75_st50_gcn_mmg gcn Mmgnet ./config/mmgnet.json 0.5 st /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/param25_unst75_gcn_mmg

# run_command prune param75_unst75_gcn_mmg gcn Mmgnet ./config/mmgnet.json 0.75 unst /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/p_redu75
# run_command prune param75_unst75_st50_gcn_mmg gcn Mmgnet ./config/mmgnet.json 0.5 st /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/param25_unst75_gcn_mmg
## method : st, unst