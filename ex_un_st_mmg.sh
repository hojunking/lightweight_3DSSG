#!/bin/bash
run_command() {
    python -m main --mode $1 --exp $2 --part $3 --model $4 --config $5 --ratio $6
}

# Call the function with different sets of arguments
# Run 1
run_command prune un25_st_gnn25 gcn Mmgnet ./config/mmgnet.json 0.25

# Run 2
run_command prune un25_st_gnn50 gcn Mmgnet ./config/mmgnet.json 0.5

# Run 3
run_command prune un25_st_gnn75 gcn Mmgnet ./config/mmgnet.json 0.75

# Run 4
run_command prune un50_st_gnn25 gcn Mmgnet ./config/mmgnet.json 0.25

run_command prune un50_st_gnn50 gcn Mmgnet ./config/mmgnet.json 0.5

run_command prune un50_st_gnn75 gcn Mmgnet ./config/mmgnet.json 0.75
