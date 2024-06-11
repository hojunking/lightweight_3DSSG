#!/bin/bash
run_command() {
    python -m main --mode $1 --exp $2 --part $3 --model $4 --config $5 --ratio $6
}

# Call the function with different sets of arguments
# Run 1
#run_command prune test gcn Mmgnet ./config/mmgnet.json 0.25

# run_command prune un75_st25_gnn gcn Mmgnet ./config/mmgnet.json 0.25

# run_command prune un75_st50_gnn gcn Mmgnet ./config/mmgnet.json 0.5

# run_command prune un75_st75_gnn gcn Mmgnet ./config/mmgnet.json 0.75
run_command prune test gcn Mmgnet ./config/mmgnet.json 0.25

# run_command prune basline_st25_gnn gcn Mmgnet ./config/mmgnet.json 0.25
# run_command prune basline_st50_gnn gcn Mmgnet ./config/mmgnet.json 0.5
# run_command prune basline_st75_gnn gcn Mmgnet ./config/mmgnet.json 0.75


