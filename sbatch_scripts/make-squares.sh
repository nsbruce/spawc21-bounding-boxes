#! /bin/bash
#SBATCH --account=def-peterdri
#SBATCH --time=15:0:0
#SBATCH --job-name=make-squares
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --output=sbatch-make-squares.log

source ~/envs/setup/rfi.sh
source ~/envs/rfi/bin/activate

# python /home/nsbruce/RFI/spawc21-bounding-boxes/make_test_set.py \\
#  --output-dir=/home/nsbruce/RFI/spawc21-bounding-boxes/yolo-train-all-overlapping/ \\
#  --percent=1.0

python /home/nsbruce/RFI/spawc21-bounding-boxes/make_eval_set.py --output-dir=/home/nsbruce/RFI/spawc21-bounding-boxes/yolo-eval-square/