#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./out_tf_hw2d_train.%j
#SBATCH -e ./err_tf_hw2d_train.%j
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J tf_hw2d_train
#
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
##SBATCH --partition=gpudev
#
# --- default case: use a single GPU on a shared node ---
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125000
#
# --- uncomment to use 2 GPUs on a shared node ---
# #SBATCH --gres=gpu:a100:2
# #SBATCH --cpus-per-task=36
# #SBATCH --mem=250000
#
# --- uncomment to use 4 GPUs on a full node ---
# #SBATCH --gres=gpu:a100:4
# #SBATCH --cpus-per-task=72
# #SBATCH --mem=500000
#
#SBATCH --time=16:00:00
##SBATCH --time=00:15:00

module purge
conda deactivate
#conda deactivate
#conda deactivate


module load anaconda/3/2023.03
module load tensorflow/gpu-cuda-12.1/2.14.0

module load keras/2.14.0

module load cupy/12.1

#module load gcc/10

conda activate conda3.10.9_tensorrt

python -V
module list
conda info --envs

echo $LD_LIBRARY_PATH

# Important:
# Set the number of OMP threads *per process* to avoid overloading of the node!
#export OMP_NUM_THREADS=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Use the environment variable SLURM_CPUS_PER_TASK to have multiprocessing
# spawn exactly as many processes as you have CPUs available.
which python
#srun python psil_hw2d_training_skeleton.py --train_dataset_path="['FFT_1800_512_c5_CPUnumba_dt01_nu5e-9_1_dataset_s64_t1','FFT_1800_512_c5_CPUnumba_dt01_nu5e-9_2_dataset_s64_t1','FFT_1800_512_c5_CPUnumba_dt01_nu5e-9_3_dataset_s64_t1']" --valid_dataset_path="['FFT_1800_512_c5_CPUnumba_dt01_nu5e-9_4_dataset_s64_t1']" --test_dataset_path="['FFT_1800_512_c5_CPUnumba_dt01_nu5e-9_5_dataset_s64_t1']" --output_path="c5HPS64_" --unrolled_steps=8 --solver_name="rk4" --poisson_method="fourier" --epochs_steps=1000 --model_type="resnet" --layers=10 --filters=32 --lr=TEMPLATE_LR --optimizer_str="adamW" --rho=TEMPLATE_RHO --SAM_warmup=0 --temporal_causality_eps=0.0 --decay_epochs=0 --batch_size_start=20 --batch_size_step=5 --batch_size_end=20 --weight_average=0 --nu=0.0 --batch_norm=True --valid_batch=1 --valid_skips=4 --valid_warmup=900 --steps_simulation=20000 --n_blocks_invariants=4 --early_stopping_patience=99 --c1="{'train':[5.0,5.0,5.0],'valid':[5.0],'test':[5.0]}" --time_window_to_keep=500 --evaluation_properties="['gamma_n_spectral','enstrophy','kinetic_energy','thermal_energy']" --tensorrt_inference=True
#srun python psil_hw2d_training_skeleton.py --train_dataset_path="['FFT_1000_512_c1_CPUnumba_dt01_nu5e-9_1_dataset_s64_t1','FFT_1000_512_c1_CPUnumba_dt01_nu5e-9_2_dataset_s64_t1','FFT_1000_512_c1_CPUnumba_dt01_nu5e-9_3_dataset_s64_t1']" --valid_dataset_path="['FFT_1000_512_c1_CPUnumba_dt01_nu5e-9_4_dataset_s64_t1']" --test_dataset_path="['FFT_1000_512_c1_CPUnumba_dt01_nu5e-9_5_dataset_s64_t1']" --output_path="NOGRADc1HPS64_" --unrolled_steps=8 --solver_name="rk4" --poisson_method="fourier" --epochs_steps=1000 --model_type="resnet" --layers=10 --filters=32 --lr=TEMPLATE_LR --optimizer_str="adamW" --rho=TEMPLATE_RHO --SAM_warmup=0 --temporal_causality_eps=0.0 --decay_epochs=0 --batch_size_start=20 --batch_size_step=5 --batch_size_end=20 --weight_average=0 --nu=0.0 --batch_norm=True --valid_batch=1 --valid_skips=4 --valid_warmup=900 --steps_simulation=20000 --n_blocks_invariants=4 --early_stopping_patience=99 --c1="{'train':[1.0,1.0,1.0],'valid':[1.0],'test':[1.0]}" --time_window_to_keep=500 --evaluation_properties="['gamma_n_spectral','enstrophy','kinetic_energy','thermal_energy']" --tensorrt_inference=True
#srun python psil_hw2d_training_skeleton.py --train_dataset_path="['FFT_1000_512_c02_CPUnumba_dt01_nu5e-9_1_dataset_s64_t1','FFT_1000_512_c02_CPUnumba_dt01_nu5e-9_2_dataset_s64_t1','FFT_1000_512_c02_CPUnumba_dt01_nu5e-9_3_dataset_s64_t1']" --valid_dataset_path="['FFT_1000_512_c02_CPUnumba_dt01_nu5e-9_4_dataset_s64_t1']" --test_dataset_path="['FFT_1000_512_c02_CPUnumba_dt01_nu5e-9_5_dataset_s64_t1']" --output_path="c02HPS64_" --unrolled_steps=8 --solver_name="rk4" --poisson_method="fourier" --epochs_steps=1000 --model_type="resnet" --layers=10 --filters=32 --lr=TEMPLATE_LR --optimizer_str="adamW" --rho=TEMPLATE_RHO --SAM_warmup=0 --temporal_causality_eps=0.0 --decay_epochs=0 --batch_size_start=20 --batch_size_step=5 --batch_size_end=20 --weight_average=0 --nu=0.0 --batch_norm=True --valid_batch=1 --valid_skips=4 --valid_warmup=900 --steps_simulation=20000 --n_blocks_invariants=4 --early_stopping_patience=99 --c1="{'train':[0.2,0.2,0.2],'valid':[0.2],'test':[0.2]}" --time_window_to_keep=500 --evaluation_properties="['gamma_n_spectral','enstrophy','kinetic_energy','thermal_energy']" --tensorrt_inference=True
#srun python psil_hw2d_training_skeleton.py --train_dataset_path="['FFT_1000_512_c02_CPUnumba_dt01_nu5e-9_1_dataset_s64_t1','FFT_1800_512_c5_CPUnumba_dt01_nu5e-9_2_dataset_s64_t1']" --valid_dataset_path="['FFT_1000_512_c02_CPUnumba_dt01_nu5e-9_4_dataset_s64_t1','FFT_1800_512_c5_CPUnumba_dt01_nu5e-9_4_dataset_s64_t1']" --test_dataset_path="['FFT_1000_512_c02_CPUnumba_dt01_nu5e-9_5_dataset_s64_t1','FFT_1800_512_c5_CPUnumba_dt01_nu5e-9_5_dataset_s64_t1']" --output_path="c02HPS64_" --unrolled_steps=8 --solver_name="rk4" --poisson_method="fourier" --epochs_steps=1000 --model_type="resnet" --layers=10 --filters=32 --lr=TEMPLATE_LR --optimizer_str="adamW" --rho=TEMPLATE_RHO --SAM_warmup=0 --temporal_causality_eps=0.0 --decay_epochs=0 --batch_size_start=20 --batch_size_step=5 --batch_size_end=20 --weight_average=0 --nu=0.0 --batch_norm=True --valid_batch=1 --valid_skips=4 --valid_warmup=900 --steps_simulation=20000 --n_blocks_invariants=4 --early_stopping_patience=20 --c1="{'train':[0.2,5.0],'valid':[0.2,5.0],'test':[0.2,5.0]}" --time_window_to_keep=500 --evaluation_properties="['gamma_n_spectral','enstrophy','kinetic_energy','thermal_energy']" --tensorrt_inference=True
#srun python psil_hw2d_training_skeleton.py --train_dataset_path="['FFT_1000_512_c02_CPUnumba_dt01_nu5e-9_1_dataset_s64_t1','FFT_1000_512_c1_CPUnumba_dt01_nu5e-9_2_dataset_s64_t1']" --valid_dataset_path="['FFT_1000_512_c02_CPUnumba_dt01_nu5e-9_4_dataset_s64_t1','FFT_1000_512_c1_CPUnumba_dt01_nu5e-9_4_dataset_s64_t1']" --test_dataset_path="['FFT_1000_512_c02_CPUnumba_dt01_nu5e-9_5_dataset_s64_t1','FFT_1000_512_c1_CPUnumba_dt01_nu5e-9_5_dataset_s64_t1']" --output_path="c021HPS64_" --unrolled_steps=8 --solver_name="rk4" --poisson_method="fourier" --epochs_steps=1000 --model_type="resnet" --layers=10 --filters=32 --lr=TEMPLATE_LR --optimizer_str="adamW" --rho=TEMPLATE_RHO --SAM_warmup=0 --temporal_causality_eps=0.0 --decay_epochs=0 --batch_size_start=20 --batch_size_step=5 --batch_size_end=20 --weight_average=0 --nu=0.0 --batch_norm=True --valid_batch=1 --valid_skips=4 --valid_warmup=900 --steps_simulation=20000 --n_blocks_invariants=4 --early_stopping_patience=20 --c1="{'train':[0.2,1.0],'valid':[0.2,1.0],'test':[0.2,1.0]}" --time_window_to_keep=500 --evaluation_properties="['gamma_n_spectral','enstrophy','kinetic_energy','thermal_energy']" --tensorrt_inference=True
srun python psil_hw2d_training_skeleton.py --train_dataset_path="['FFT_1000_512_c1_CPUnumba_dt01_nu5e-9_1_dataset_s64_t1','FFT_1800_512_c5_CPUnumba_dt01_nu5e-9_2_dataset_s64_t1']" --valid_dataset_path="['FFT_1000_512_c1_CPUnumba_dt01_nu5e-9_4_dataset_s64_t1','FFT_1800_512_c5_CPUnumba_dt01_nu5e-9_4_dataset_s64_t1']" --test_dataset_path="['FFT_1000_512_c1_CPUnumba_dt01_nu5e-9_5_dataset_s64_t1','FFT_1800_512_c5_CPUnumba_dt01_nu5e-9_5_dataset_s64_t1']" --output_path="c15HPS64_" --unrolled_steps=8 --solver_name="rk4" --poisson_method="fourier" --epochs_steps=1000 --model_type="resnet" --layers=10 --filters=32 --lr=TEMPLATE_LR --optimizer_str="adamW" --rho=TEMPLATE_RHO --SAM_warmup=0 --temporal_causality_eps=0.0 --decay_epochs=0 --batch_size_start=20 --batch_size_step=5 --batch_size_end=20 --weight_average=0 --nu=0.0 --batch_norm=True --valid_batch=1 --valid_skips=4 --valid_warmup=900 --steps_simulation=20000 --n_blocks_invariants=4 --early_stopping_patience=20 --c1="{'train':[1.0,5.0],'valid':[1.0,5.0],'test':[1.0,5.0]}" --time_window_to_keep=500 --evaluation_properties="['gamma_n_spectral','enstrophy','kinetic_energy','thermal_energy']" --tensorrt_inference=True
