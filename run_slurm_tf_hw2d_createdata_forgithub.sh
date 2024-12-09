#!/bin/bash -l
#
#SBATCH -o ./out_tf_hw2d.%j
#SBATCH -e ./err_tf_hw2d.%j
#SBATCH -D ./
#SBATCH -J tf_hw2d
#SBATCH --nodes=1             # request a full node
#SBATCH --ntasks-per-node=1   # only start 1 task via srun because Python multiprocessing starts more tasks internally
#SBATCH --cpus-per-task=72    # assign all the cores to that first task to make room for Python's multiprocessing tasks
#SBATCH --time=09:10:00

module purge

module load gcc/12
module load anaconda/3/2023.03
module load protobuf/4.22.3

module load tensorboard/2.12.0
#CPU
module load tensorflow/cpu/2.12.0       #(after loading anaconda/3/2023.03)  #Python 3.8–3.11
module load keras/2.12.0

module load ffmpeg/4.4

python -V
#conda init bash
conda activate py310_
python -V


# Important:
# Set the number of OMP threads *per process* to avoid overloading of the node!
export OMP_NUM_THREADS=1

srun python src/tf_hw2d/subsampling/create_dataset.py --original_path="FFT_1800_512_c5_CPUnumba_dt01_nu5e-9_5" --new_path="FFT_1800_512_c5_CPUnumba_dt01_nu5e-9_5_dataset_s64_t1" --spatial_downsampling_ratio=8.0 --time_downsampling_ratio=1 --sampling_method="fourier"







