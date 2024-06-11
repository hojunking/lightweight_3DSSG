#!/bin/bash
run_command() {
    python -m main --mode $1 --exp $2 --part $3 --model $4 --config $5 --ratio $6
}

# Call the function with different sets of arguments
# Run 1
run_command train KD_VLSAT_SGPN gcn sgpn ./config/mmgnet.json 0.25


