#!/bin/bash
run_command() {
    python -m main --mode $1 --exp $2 --part $3 --model $4 --config $5
}

# Call the function with different sets of arguments

# Run 1
run_command prune prune encoder sgfn ./config/mmgnet.json

# Run 2
run_command prune prune gcn sgfn ./config/mmgnet.json

# Run 3
run_command train prune classifier sgfn ./config/mmgnet.json

# Run 4
run_command train prune all sgfn ./config/mmgnet.json