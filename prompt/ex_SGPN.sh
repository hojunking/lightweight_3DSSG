 #!/bin/bash
run_command() {
    python -m main --mode $1 --exp $2 --part $3 --config $4 --st_ratio $5 --unst_ratio $6 --pretrained $7

}
run_command prune test gnn ./config/SGPN.json 0.1 0 x

# run_command train param40_num_block3_gcn_sgpn gcn ./config/SGPN.json 0 0 x
# run_command prune param40_num_block3_25_gcn_sgpn gcn ./config/SGPN.json 0.25 0 /home/knuvki/Desktop/song/VLSAT_pruning/config/ckp/SGPN/param40_num_block3_gcn_sgpn
# run_command prune param40_num_block3_50_gcn_sgpn gcn ./config/SGPN.json 0.5 0 /home/knuvki/Desktop/song/VLSAT_pruning/config/ckp/SGPN/param40_num_block3_gcn_sgpn
# run_command prune param40_num_block3_75_gcn_sgpn gcn ./config/SGPN.json 0.75 0 /home/knuvki/Desktop/song/VLSAT_pruning/config/ckp/SGPN/param40_num_block3_gcn_sgpn
# run_command eval eval_sgpn_baseline_sgpn gcn ./config/SGPN.json 0 0 /home/knuvki/Desktop/song/VLSAT_pruning/config/ckp/SGPN/real_baseline_sgpn
# ## st pruning inference 5 - 75
# run_command eval eval_param40_st05_gcn_sgpn gcn ./config/SGPN.json 0.05 0 /home/knuvki/Desktop/song/VLSAT_pruning/config/ckp/SGPN/param40_num_block3_gcn_sgpn
# run_command eval eval_param40_st10_gcn_sgpn gcn ./config/SGPN.json 0.1 0 /home/knuvki/Desktop/song/VLSAT_pruning/config/ckp/SGPN/param40_num_block3_gcn_sgpn
# run_command eval eval_param40_st15_gcn_sgpn gcn ./config/SGPN.json 0.15 0 /home/knuvki/Desktop/song/VLSAT_pruning/config/ckp/SGPN/param40_num_block3_gcn_sgpn
# run_command eval eval_param40_st20_gcn_sgpn gcn ./config/SGPN.json 0.20 0 /home/knuvki/Desktop/song/VLSAT_pruning/config/ckp/SGPN/param40_num_block3_gcn_sgpn
# run_command eval eval_param40_st25_gcn_sgpn gcn ./config/SGPN.json 0.25 0 /home/knuvki/Desktop/song/VLSAT_pruning/config/ckp/SGPN/param40_num_block3_gcn_sgpn
# run_command eval eval_param40_st30_gcn_sgpn gcn ./config/SGPN.json 0.30 0 /home/knuvki/Desktop/song/VLSAT_pruning/config/ckp/SGPN/param40_num_block3_gcn_sgpn
# run_command eval eval_param40_st35_gcn_sgpn gcn ./config/SGPN.json 0.35 0 /home/knuvki/Desktop/song/VLSAT_pruning/config/ckp/SGPN/param40_num_block3_gcn_sgpn
# run_command eval eval_param40_st40_gcn_sgpn gcn ./config/SGPN.json 0.40 0 /home/knuvki/Desktop/song/VLSAT_pruning/config/ckp/SGPN/param40_num_block3_gcn_sgpn
# run_command eval eval_param40_st45_gcn_sgpn gcn ./config/SGPN.json 0.45 0 /home/knuvki/Desktop/song/VLSAT_pruning/config/ckp/SGPN/param40_num_block3_gcn_sgpn
# run_command eval eval_param40_st50_gcn_sgpn gcn ./config/SGPN.json 0.50 0 /home/knuvki/Desktop/song/VLSAT_pruning/config/ckp/SGPN/param40_num_block3_gcn_sgpn
# run_command eval eval_param40_st55_gcn_sgpn gcn ./config/SGPN.json 0.55 0 /home/knuvki/Desktop/song/VLSAT_pruning/config/ckp/SGPN/param40_num_block3_gcn_sgpn
# run_command eval eval_param40_st60_gcn_sgpn gcn ./config/SGPN.json 0.60 0 /home/knuvki/Desktop/song/VLSAT_pruning/config/ckp/SGPN/param40_num_block3_gcn_sgpn
# run_command eval eval_param40_st65_gcn_sgpn gcn ./config/SGPN.json 0.65 0 /home/knuvki/Desktop/song/VLSAT_pruning/config/ckp/SGPN/param40_num_block3_gcn_sgpn
# run_command eval eval_param40_st70_gcn_sgpn gcn ./config/SGPN.json 0.70 0 /home/knuvki/Desktop/song/VLSAT_pruning/config/ckp/SGPN/param40_num_block3_gcn_sgpn
# run_command eval eval_param40_st75_gcn_sgpn gcn ./config/SGPN.json 0.75 0 /home/knuvki/Desktop/song/VLSAT_pruning/config/ckp/SGPN/param40_num_block3_gcn_sgpn
# ## st pruning inference 5 - 75
# ## unst pruning inference 5 - 75
# run_command eval eval_sgpn_unst05_gcn_sgpn gcn ./config/SGPN.json 0 0.05 /home/knuvki/Desktop/song/VLSAT_pruning/config/ckp/SGPN/real_baseline_sgpn
# run_command eval eval_sgpn_unst10_gcn_sgpn gcn ./config/SGPN.json 0 0.10 /home/knuvki/Desktop/song/VLSAT_pruning/config/ckp/SGPN/real_baseline_sgpn
# run_command eval eval_sgpn_unst15_gcn_sgpn gcn ./config/SGPN.json 0 0.15 /home/knuvki/Desktop/song/VLSAT_pruning/config/ckp/SGPN/real_baseline_sgpn
# run_command eval eval_sgpn_unst20_gcn_sgpn gcn ./config/SGPN.json 0 0.20 /home/knuvki/Desktop/song/VLSAT_pruning/config/ckp/SGPN/real_baseline_sgpn
# run_command eval eval_sgpn_unst25_gcn_sgpn gcn ./config/SGPN.json 0 0.25 /home/knuvki/Desktop/song/VLSAT_pruning/config/ckp/SGPN/real_baseline_sgpn
# run_command eval eval_sgpn_unst30_gcn_sgpn gcn ./config/SGPN.json 0 0.30 /home/knuvki/Desktop/song/VLSAT_pruning/config/ckp/SGPN/real_baseline_sgpn
# run_command eval eval_sgpn_unst35_gcn_sgpn gcn ./config/SGPN.json 0 0.35 /home/knuvki/Desktop/song/VLSAT_pruning/config/ckp/SGPN/real_baseline_sgpn
# run_command eval eval_sgpn_unst40_gcn_sgpn gcn ./config/SGPN.json 0 0.40 /home/knuvki/Desktop/song/VLSAT_pruning/config/ckp/SGPN/real_baseline_sgpn
# run_command eval eval_sgpn_unst45_gcn_sgpn gcn ./config/SGPN.json 0 0.45 /home/knuvki/Desktop/song/VLSAT_pruning/config/ckp/SGPN/real_baseline_sgpn
# run_command eval eval_sgpn_unst50_gcn_sgpn gcn ./config/SGPN.json 0 0.50 /home/knuvki/Desktop/song/VLSAT_pruning/config/ckp/SGPN/real_baseline_sgpn
# run_command eval eval_sgpn_unst55_gcn_sgpn gcn ./config/SGPN.json 0 0.55 /home/knuvki/Desktop/song/VLSAT_pruning/config/ckp/SGPN/real_baseline_sgpn
# run_command eval eval_sgpn_unst60_gcn_sgpn gcn ./config/SGPN.json 0 0.60 /home/knuvki/Desktop/song/VLSAT_pruning/config/ckp/SGPN/real_baseline_sgpn
# run_command eval eval_sgpn_unst65_gcn_sgpn gcn ./config/SGPN.json 0 0.65 /home/knuvki/Desktop/song/VLSAT_pruning/config/ckp/SGPN/real_baseline_sgpn
# run_command eval eval_sgpn_unst70_gcn_sgpn gcn ./config/SGPN.json 0 0.70 /home/knuvki/Desktop/song/VLSAT_pruning/config/ckp/SGPN/real_baseline_sgpn
# run_command eval eval_sgpn_unst75_gcn_sgpn gcn ./config/SGPN.json 0 0.75 /home/knuvki/Desktop/song/VLSAT_pruning/config/ckp/SGPN/real_baseline_sgpn