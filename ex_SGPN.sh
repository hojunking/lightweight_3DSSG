 #!/bin/bash
run_command() {
    python -m main --mode $1 --exp $2 --part $3 --model $4 --config $5
}

# Call the function with different sets of arguments

# Run 1
#run_command prune st_enco_prune50 encoder sgpn ./config/mmgnet.json

# Run 2
#run_command prune st_gcn_prune50 gcn sgpn ./config/mmgnet.json

# Run 3
run_command prune st_classif_prune50 classifier sgpn ./config/mmgnet.json

# Run 4
run_command prune st_all_prune50 all sgpn ./config/mmgnet.json
