#!/bin/bash
run_command() {
    python -m main --mode $1 --exp $2 --part $3 --config $4 --st_ratio $5 --unst_ratio $6 --pretrained $7

}
#run_command train test gcn ./config/cfg_files/mmgnet_edge_dim256.json 0 0 x
#run_command train redu_edge_dim256_gcn_mmg gcn ./config/mmgnet_edge_dim256.json 0 0 x

#run_command prune param65_st35_gcn_mmg gcn ./config/mmgnet_redu.json 0.35 0 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/param75_edgedim_attndim_gcn_mmg
#run_command train all256_gcn_mmg gcn ./config/mmgnet.json 0 0 x
#run_command eval test gcn ./config/mmgnet.json 0.80 0 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/vlsat_baseline

# run_command train param75_edgedim_attndim_gcn_mmg gcn ./config/mmgnet_redu.json 0 0 x
# run_command prune param75_st25_gcn_mmg gcn ./config/mmgnet_redu.json 0.25 0 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/param75_edgedim_attndim_gcn_mmg
# run_command prune param75_st55_gcn_mmg gcn ./config/mmgnet_redu.json 0.5 0 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/param75_edgedim_attndim_gcn_mmg

# run_command eval eval_param50_2mlp_st05_gcn_mmg gcn ./config/mmgnet.json 0.05 0 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/param50_edge_mlp_reduction_gcn_mmg
# run_command eval eval_param50_2mlp_st10_gcn_mmg gcn ./config/mmgnet.json 0.1 0 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/param50_edge_mlp_reduction_gcn_mmg
# run_command eval eval_param50_2mlp_st15_gcn_mmg gcn ./config/mmgnet.json 0.15 0 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/param50_edge_mlp_reduction_gcn_mmg
# run_command eval eval_param50_2mlp_st20_gcn_mmg gcn ./config/mmgnet.json 0.20 0 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/param50_edge_mlp_reduction_gcn_mmg
# run_command eval eval_param50_2mlp_st25_gcn_mmg gcn ./config/mmgnet.json 0.25 0 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/param50_edge_mlp_reduction_gcn_mmg

# run_command eval eval_mmg_baseline_mmg gcn ./config/mmgnet.json 0 0 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/vlsat_baseline
# ## st pruning inference 5 - 95
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
run_command eval eval_mmg_st80_gcn_mmg gcn ./config/mmgnet.json 0.80 0 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/vlsat_baseline
run_command eval eval_mmg_st85_gcn_mmg gcn ./config/mmgnet.json 0.85 0 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/vlsat_baseline
run_command eval eval_mmg_st90_gcn_mmg gcn ./config/mmgnet.json 0.90 0 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/vlsat_baseline
run_command eval eval_mmg_st95_gcn_mmg gcn ./config/mmgnet.json 0.95 0 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/vlsat_baseline
# ## unst pruning inference 5 - 95
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
run_command eval eval_mmg_unst80_gcn_mmg gcn ./config/mmgnet.json 0 0.80 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/vlsat_baseline
run_command eval eval_mmg_unst85_gcn_mmg gcn ./config/mmgnet.json 0 0.85 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/vlsat_baseline
run_command eval eval_mmg_unst90_gcn_mmg gcn ./config/mmgnet.json 0 0.90 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/vlsat_baseline
run_command eval eval_mmg_unst95_gcn_mmg gcn ./config/mmgnet.json 0 0.95 /home/knuvi/Desktop/song/lightweight_3DSSG/config/ckp/Mmgnet/vlsat_baseline