#! /bin/bash
#SBATCH --account=def-peterdri
#SBATCH --time=5:0:0
#SBATCH --job-name=post-process-rfinder
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --output=sbatch-post-process-rfinder.log
#SBATCH --gres=gpu:2

# source ~/envs/setup/darknet.sh
source ~/envs/setup/rfi.sh
source ~/envs/rfi/bin/activate

python /home/nsbruce/RFI/spawc21-bounding-boxes/combine-predictions.py --pred-dir=/home/nsbruce/RFI/spawc21-bounding-boxes/yolo-test-square/Images --bounds-file=/home/nsbruce/RFI/spawc21-bounding-boxes/yolo-test-square/bounds.pkl --output-dir=/home/nsbruce/RFI/spawc21-bounding-boxes/yolo-test-square/preds1
