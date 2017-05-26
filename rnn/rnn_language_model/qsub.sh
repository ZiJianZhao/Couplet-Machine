qsub -cwd -S /bin/bash -o LOG -e ERR -l hostname=viterbi,gpu=1 -q GTX1070.q run.sh
