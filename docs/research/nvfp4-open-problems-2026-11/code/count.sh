#!/bin/bash
set -e
D=$(dirname "$0")
cd "$D"
export PATH=/usr/local/cuda-13.0/bin:$PATH
nvcc --version | tail -2
nvcc -O3 -arch=sm_120a -cubin -o sr.cubin sr_variants.cu
cuobjdump -sass sr.cubin > sr.sass
# count SASS instructions per kernel (exclude NOP and trailing BRA loop)
awk '
/Function : / {if (name!="") printf "%-28s %4d total  %4d int-mul(IMAD*/IMUL)  %4d lop3/shf/sgxt\n", name, n, m, l; name=$3; n=0; m=0; l=0; next}
/^ *\/\*[0-9a-f]+\*\// {
  line=$0; sub(/^ *\/\*[0-9a-f]+\*\/ */,"",line); split(line,a," "); op=a[1]; if (op ~ /^@/) op=a[2];
  if (op ~ /^NOP/ ) next; if (op=="BRA" && line ~ /BRA `\(\.L_x_[0-9]+\)/ && prev=="EXIT") next;
  n++; if (op ~ /^IMAD/ && op !~ /^IMAD\.MOV/ && op !~ /^IMAD\.IADD/ && op !~ /^IMAD\.SHL/ || op ~ /^IMUL/) m++;
  if (op ~ /^(LOP3|SHF|SGXT|PRMT|LEA)/) l++; prev=op}
END {printf "%-28s %4d total  %4d int-mul(IMAD*/IMUL)  %4d lop3/shf/sgxt\n", name, n, m, l}' sr.sass
