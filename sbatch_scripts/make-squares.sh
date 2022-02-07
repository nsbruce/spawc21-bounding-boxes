#! /bin/bash
#SBATCH --account=def-peterdri
#SBATCH --time=15:0:0
#SBATCH --job-name=make-training-squares-grey
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --output=sbatch-make-training-squares-grey.log

source ~/envs/setup/rfi.sh
source ~/envs/rfi/bin/activate

# SHOULD WORK WITH SBATCH --time=15:0:0
# python /home/nsbruce/RFI/spawc21-bounding-boxes/make_test_set.py --output-dir=/home/nsbruce/RFI/spawc21-bounding-boxes/yolo-train-all-overlapping/ --percent=1.0
# python /home/nsbruce/RFI/spawc21-bounding-boxes/make_test_set.py --output-dir=/home/nsbruce/RFI/spawc21-bounding-boxes/results/train-median-normalized --percent=1.0 --skip-empty
# python /home/nsbruce/RFI/spawc21-bounding-boxes/make_test_set.py --output-dir=/home/nsbruce/RFI/spawc21-bounding-boxes/results/train-median-normalized-skip-empty --percent=1.0 --skip-empty

# python /home/nsbruce/RFI/spawc21-bounding-boxes/make_eval_set.py --output-dir=/home/nsbruce/RFI/spawc21-bounding-boxes/yolo-eval-square/
# Make greysscale eval squares
# python /home/nsbruce/RFI/spawc21-bounding-boxes/make_eval_set.py --output-dir=/home/nsbruce/RFI/spawc21-bounding-boxes/results/eval-median-normalized
# python /home/nsbruce/RFI/ursi-data-processing/ursi-timeseries-to-images.py --output-dir=/home/nsbruce/RFI/ursi-data-processing/results/1024-greyscale --input-dir=/home/nsbruce/RFI/ursi-data-processing/sigmfs

# python /home/nsbruce/RFI/ursi-data-processing/h5-to-images.py --output-dir=/home/nsbruce/RFI/ursi-data-processing/results/1024-h5 --h5-file=/home/nsbruce/RFI/ursi-data-processing/sensible-rfind.h5



# seff:
    # Job ID: 11998483
    # Cluster: cedar
    # User/Group: nsbruce/nsbruce
    # State: COMPLETED (exit code 0)
    # Nodes: 1
    # Cores per node: 6
    # CPU Utilized: 04:20:34
    # CPU Efficiency: 15.92% of 1-03:16:24 core-walltime
    # Job Wall-clock time: 04:32:44
    # Memory Utilized: 4.21 GB
    # Memory Efficiency: 13.16% of 32.00 GB
# input:
    # --time=5:0:0
    # --mem=32G
    # --cpus-per-task=6

# python /home/nsbruce/RFI/spawc21-bounding-boxes/make_test_set.py --skip-empty --output-dir=/home//nsbruce/RFI/spawc21-bounding-boxes/results/train-zero-mean-clipped 




#seff:
    # Job ID: 12008171
    # Cluster: cedar
    # User/Group: nsbruce/nsbruce
    # State: TIMEOUT (exit code 0)
    # Nodes: 1
    # Cores per node: 6
    # CPU Utilized: 04:55:25
    # CPU Efficiency: 16.41% of 1-06:00:42 core-walltime
    # Job Wall-clock time: 05:00:07
    # Memory Utilized: 2.26 GB
    # Memory Efficiency: 7.06% of 32.00 GB
# input:
    # --time=5:0:0
    # --mem=32G
    # --cpus-per-task=6
python /home/nsbruce/RFI/spawc21-bounding-boxes/make_test_set.py --skip-empty --output-dir=/home//nsbruce/RFI/spawc21-bounding-boxes/results/train-zero-mean-clipped-greyscale --greyscale 
