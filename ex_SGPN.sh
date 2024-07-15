 #!/bin/bash
run_command() {
    python -m main --mode $1 --exp $2 --part $3 --model $4 --config $5 --ratio $6 --method $7 --pretrained $8

}
run_command train real_sgpn gcn sgpn ./config/SGPN.json 0 st x