 #!/bin/bash
run_command() {
    python -m main --mode $1 --exp $2 --part $3 --config $4 --st_ratio $5 --unst_ratio $6 --pretrained $7

}
run_command train real_sgpn gcn ./config/SGPN.json 0 st x