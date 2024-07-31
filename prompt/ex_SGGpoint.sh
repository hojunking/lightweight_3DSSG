#!/bin/bash
run_command() {
    python -m main --mode $1 --exp $2 --part $3 --config $4 --st_ratio $5 --unst_ratio $6 --pretrained $7
}

run_command train SGGpoint_vlsat_pointnet gcn ./config/SGGpoint.json 0 0 x
