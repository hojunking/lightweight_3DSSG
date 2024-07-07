#!/bin/bash
run_command() {
    python -m main --mode $1 --exp $2 --part $3 --model $4 --config $5 --ratio $6 --method $7
}

# Call the function with different sets of arguments

# Run 1
#run_command prune st_enco_prune50 encoder sgfn ./config/mmgnet.json

# Run 2
#run_command prune st_gcn_prune50 gcn sgfn ./config/mmgnet.json

# Run 3
#run_command prune st_classif_prune50 classifier sgfn ./config/mmgnet.json

# Run 4
#run_command prune st_all_prune50 all sgfn ./config/mmgnet.json

run_command prune test gcn sgfn ./config/mmgnet.json 0.5 unst
