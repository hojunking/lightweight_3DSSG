#!/bin/bash
run_command() {
    python -m main --mode $1 --exp $2 --part $3 --config $4 --st_ratio $5 --unst_ratio $6 --pretrained $7

}
#run_command eval test gcn ./config/mmgnet.json 0 0 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/vlsat_baseline
#run_command train param50_real_mmg gcn ./config/mmgnet.json 0 0 x
run_command prune param50_st35_gcn_real_mmg gcn ./config/mmgnet.json 0.35 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/p_redu50
run_command prune param50_st45_gcn_real_mmg gcn ./config/mmgnet.json 0.45 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/p_redu50

#run_command prune param50_st25_gcn_real_mmg gcn ./config/mmgnet.json 0.25 0 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/param50_real_mmg

# run_command train param50_2mlp_reduction_gcn_mmg gcn ./config/mmgnet.json 0 0 x
# run_command eval eval_param50_2mlp_st05_gcn_mmg gcn ./config/mmgnet.json 0.05 0 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/param50_2mlp_reduction_gcn_mmg
# run_command eval eval_param50_2mlp_st10_gcn_mmg gcn ./config/mmgnet.json 0.1 0 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/param50_2mlp_reduction_gcn_mmg
# run_command eval eval_param50_2mlp_st15_gcn_mmg gcn ./config/mmgnet.json 0.15 0 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/param50_2mlp_reduction_gcn_mmg
# run_command eval eval_param50_2mlp_st20_gcn_mmg gcn ./config/mmgnet.json 0.20 0 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/param50_2mlp_reduction_gcn_mmg
# run_command eval eval_param50_2mlp_st25_gcn_mmg gcn ./config/mmgnet.json 0.25 0 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/param50_2mlp_reduction_gcn_mmg

# run_command eval eval_mmg_baseline_mmg gcn ./config/mmgnet.json 0 0 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/vlsat_baseline
# ## st pruning inference 5 - 75
# run_command eval eval_mmg_st05_gcn_mmg gcn ./config/mmgnet.json 0.05 0 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/vlsat_baseline
# run_command eval eval_mmg_st10_gcn_mmg gcn ./config/mmgnet.json 0.1 0 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/vlsat_baseline
# run_command eval eval_mmg_st15_gcn_mmg gcn ./config/mmgnet.json 0.15 0 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/vlsat_baseline
# run_command eval eval_mmg_st20_gcn_mmg gcn ./config/mmgnet.json 0.20 0 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/vlsat_baseline
# run_command eval eval_mmg_st25_gcn_mmg gcn ./config/mmgnet.json 0.25 0 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/vlsat_baseline
# run_command eval eval_mmg_st30_gcn_mmg gcn ./config/mmgnet.json 0.30 0 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/vlsat_baseline
# run_command eval eval_mmg_st35_gcn_mmg gcn ./config/mmgnet.json 0.35 0 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/vlsat_baseline
# run_command eval eval_mmg_st40_gcn_mmg gcn ./config/mmgnet.json 0.40 0 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/vlsat_baseline
# run_command eval eval_mmg_st45_gcn_mmg gcn ./config/mmgnet.json 0.45 0 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/vlsat_baseline
# run_command eval eval_mmg_st50_gcn_mmg gcn ./config/mmgnet.json 0.50 0 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/vlsat_baseline
# run_command eval eval_mmg_st55_gcn_mmg gcn ./config/mmgnet.json 0.55 0 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/vlsat_baseline
# run_command eval eval_mmg_st60_gcn_mmg gcn ./config/mmgnet.json 0.60 0 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/vlsat_baseline
# run_command eval eval_mmg_st65_gcn_mmg gcn ./config/mmgnet.json 0.65 0 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/vlsat_baseline
# run_command eval eval_mmg_st70_gcn_mmg gcn ./config/mmgnet.json 0.70 0 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/vlsat_baseline
# run_command eval eval_mmg_st75_gcn_mmg gcn ./config/mmgnet.json 0.75 0 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/vlsat_baseline

# ## unst pruning inference 5 - 75
# run_command eval eval_mmg_unst05_gcn_mmg gcn ./config/mmgnet.json 0 0.05 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/vlsat_baseline
# run_command eval eval_mmg_unst10_gcn_mmg gcn ./config/mmgnet.json 0 0.10 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/vlsat_baseline
# run_command eval eval_mmg_unst15_gcn_mmg gcn ./config/mmgnet.json 0 0.15 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/vlsat_baseline
# run_command eval eval_mmg_unst20_gcn_mmg gcn ./config/mmgnet.json 0 0.20 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/vlsat_baseline
# run_command eval eval_mmg_unst25_gcn_mmg gcn ./config/mmgnet.json 0 0.25 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/vlsat_baseline
# run_command eval eval_mmg_unst30_gcn_mmg gcn ./config/mmgnet.json 0 0.30 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/vlsat_baseline
# run_command eval eval_mmg_unst35_gcn_mmg gcn ./config/mmgnet.json 0 0.35 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/vlsat_baseline
# run_command eval eval_mmg_unst40_gcn_mmg gcn ./config/mmgnet.json 0 0.40 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/vlsat_baseline
# run_command eval eval_mmg_unst45_gcn_mmg gcn ./config/mmgnet.json 0 0.45 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/vlsat_baseline
# run_command eval eval_mmg_unst50_gcn_mmg gcn ./config/mmgnet.json 0 0.50 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/vlsat_baseline
# run_command eval eval_mmg_unst55_gcn_mmg gcn ./config/mmgnet.json 0 0.55 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/vlsat_baseline
# run_command eval eval_mmg_unst60_gcn_mmg gcn ./config/mmgnet.json 0 0.60 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/vlsat_baseline
# run_command eval eval_mmg_unst65_gcn_mmg gcn ./config/mmgnet.json 0 0.65 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/vlsat_baseline
# run_command eval eval_mmg_unst70_gcn_mmg gcn ./config/mmgnet.json 0 0.70 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/vlsat_baseline
# run_command eval eval_mmg_unst75_gcn_mmg gcn ./config/mmgnet.json 0 0.75 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/vlsat_baseline