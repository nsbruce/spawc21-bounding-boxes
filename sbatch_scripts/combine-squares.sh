#! /bin/bash
#SBATCH --account=def-peterdri
#SBATCH --time=5:0:0
#SBATCH --job-name=combine-squares
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --output=sbatch-combine-squares.log

source ~/envs/setup/rfi.sh
source ~/envs/rfi/bin/activate



# SHOULD WORK SBATCH --time=15:0:0
# python /home/nsbruce/projects/def-msteve/nsbruce/RFI/spawc21-bounding-boxes/combine-predictions.py --results-json=/home/nsbruce/projects/def-msteve/nsbruce/RFI/spawc21-bounding-boxes/yolo-train-all-overlapping/result.json --bounds-file=/home/nsbruce/projects/def-msteve/nsbruce/RFI/spawc21-bounding-boxes/yolo-train-all-overlapping/bounds.pkl --output-dir=/home/nsbruce/projects/def-msteve/nsbruce/RFI/spawc21-bounding-boxes/yolo-train-all-overlapping/combined/ --threshold=0.73


# Test portion of the training set
# python /home/nsbruce/projects/def-msteve/nsbruce/RFI/spawc21-bounding-boxes/combine-predictions.py \\
#  --results-json=/home/nsbruce/projects/def-msteve/nsbruce/RFI/spawc21-bounding-boxes/yolo-test-square/result.json \\
#  --bounds-file=/home/nsbruce/projects/def-msteve/nsbruce/RFI/spawc21-bounding-boxes/yolo-test-square/bounds.pkl \\
#  --output-dir=/home/nsbruce/projects/def-msteve/nsbruce/RFI/spawc21-bounding-boxes/yolo-test-square/combined2/ \\
#  --threshold=0.73 \\
#  --plot


# SHOULD WORK SBATCH --time=2:0:0
# python /home/nsbruce/projects/def-msteve/nsbruce/RFI/spawc21-bounding-boxes/combine-predictions.py --results-json=/home/nsbruce/projects/def-msteve/nsbruce/RFI/spawc21-bounding-boxes/yolo-eval-square/result.json --bounds-file=/home/nsbruce/projects/def-msteve/nsbruce/RFI/spawc21-bounding-boxes/yolo-eval-square/bounds.pkl --output-dir=/home/nsbruce/projects/def-msteve/nsbruce/RFI/spawc21-bounding-boxes/yolo-eval-square/combined/ --threshold=0.73 --eval-set --plot

python /home/nsbruce/projects/def-msteve/nsbruce/RFI/spawc21-bounding-boxes/combine-predictions.py --results-json=/home/nsbruce/projects/def-msteve/nsbruce/RFI/spawc21-bounding-boxes/yolo-eval-square/result-Aug3-new-weights.json --bounds-file=/home/nsbruce/projects/def-msteve/nsbruce/RFI/spawc21-bounding-boxes/yolo-eval-square/bounds.pkl --output-dir=/home/nsbruce/projects/def-msteve/nsbruce/RFI/spawc21-bounding-boxes/yolo-eval-square/combined-Aug3-new-weights/ --threshold=0.73 --eval-set --plot

