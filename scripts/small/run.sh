#BSUB -q gpuqueue
#BSUB -o %J.stdout
#BSUB -gpu "num=1:j_exclusive=yes"
#BSUB -R "rusage[mem=32] span[ptile=1]"
#BSUB -R A100
#BSUB -W 0:59
#BSUB -n 1

module load cuda/12.2
python run.py --model $MODEL --data $DATA