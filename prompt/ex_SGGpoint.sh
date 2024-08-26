#!/bin/bash
run_command() {
    python -m main --mode $1 --exp $2 --part $3 --config $4 --st_ratio $5 --unst_ratio $6 --pretrained $7
}

#run_command train SGGpoint_test2 gcn ./config/SGGpoint.json 0 0 x
#run_command eval eval_test gcn ./config/SGGpoint.json 0 0 /home/song/Desktop/song/lightweight_3DSSG/config/ckp/SGGpoint/SGGpoint_test
# run_command prune sggpoint_st25_gcn_sggpoint gcn ./config/SGGpoint.json 0.25 0 x
# run_command prune sggpoint_st50_gcn_sggpoint gcn ./config/SGGpoint.json 0.50 0 x
# run_command prune sggpoint_st75_gcn_sggpoint gcn ./config/SGGpoint.json 0.75 0 x

#run_command prune sggpoint_unst25_gcn_sggpoint gcn ./config/SGGpoint.json 0 0.25 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGGpoint/sggpoint_baseline
# run_command eval eval_sggpoint_unst05_all all ./config/SGGpoint.json 0 0.05 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGGpoint/sggpoint_baseline
# run_command eval eval_sggpoint_unst10_all all ./config/SGGpoint.json 0 0.10 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGGpoint/sggpoint_baseline
# run_command eval eval_sggpoint_unst15_all all ./config/SGGpoint.json 0 0.15 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGGpoint/sggpoint_baseline
# run_command eval eval_sggpoint_unst20_all all ./config/SGGpoint.json 0 0.20 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGGpoint/sggpoint_baseline
# run_command eval eval_sggpoint_unst25_all gcn ./config/SGGpoint.json 0 0.25 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGGpoint/sggpoint_baseline
# run_command eval eval_sggpoint_unst30_all gcn ./config/SGGpoint.json 0 0.30 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGGpoint/sggpoint_baseline
# run_command eval eval_sggpoint_unst35_all gcn ./config/SGGpoint.json 0 0.35 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGGpoint/sggpoint_baseline
# run_command eval eval_sggpoint_unst40_all gcn ./config/SGGpoint.json 0 0.40 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGGpoint/sggpoint_baseline
# run_command eval eval_sggpoint_unst45_all gcn ./config/SGGpoint.json 0 0.45 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGGpoint/sggpoint_baseline
#run_command eval eval_sggpoint_unst50_all all ./config/SGGpoint.json 0 0.50 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGGpoint/sggpoint_baseline
#run_command eval eval_sggpoint_unst55_all gcn ./config/SGGpoint.json 0 0.55 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGGpoint/sggpoint_baseline
#run_command eval eval_sggpoint_unst60_all all ./config/SGGpoint.json 0 0.60 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGGpoint/sggpoint_baseline
#run_command eval eval_sggpoint_unst65_all gcn ./config/SGGpoint.json 0 0.65 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGGpoint/sggpoint_baseline
#run_command eval eval_sggpoint_unst70_all all ./config/SGGpoint.json 0 0.70 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGGpoint/sggpoint_baseline
#run_command eval eval_sggpoint_unst75_all gcn ./config/SGGpoint.json 0 0.75 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGGpoint/sggpoint_baseline
#run_command eval eval_sggpoint_unst80_all all ./config/SGGpoint.json 0 0.80 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGGpoint/sggpoint_baseline
#run_command eval eval_sggpoint_unst85_all gcn ./config/SGGpoint.json 0 0.85 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGGpoint/sggpoint_baseline
#run_command eval eval_sggpoint_unst90_all all ./config/SGGpoint.json 0 0.90 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGGpoint/sggpoint_baseline

run_command eval eval_sggpoint_unst25_gcn gnn ./config/SGGpoint.json 0 0.25 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGGpoint/sggpoint_baseline
run_command eval eval_sggpoint_unst50_gcn gnn ./config/SGGpoint.json 0 0.50 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGGpoint/sggpoint_baseline

run_command eval eval_sggpoint_unst25_encoder encoder ./config/SGGpoint.json 0 0.25 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGGpoint/sggpoint_baseline
run_command eval eval_sggpoint_unst50_encoder encoder ./config/SGGpoint.json 0 0.50 /home/oi/Desktop/song/lightweight_3DSSG/config/ckp/SGGpoint/sggpoint_baseline
