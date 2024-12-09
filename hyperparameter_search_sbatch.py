import numpy as np
import scipy
from scipy.stats import qmc
import subprocess

points = 5
seed = 3
np.random.seed(seed=seed)


l_true_bounds = [1e-5, 0.005]
l_bounds = [1e-5, 0.005]
u_bounds = [5e-3, 0.1]
template_names = ["TEMPLATE_LR", "TEMPLATE_RHO"]


dimension = len(template_names)





def my_scale(sample, l_true_bounds, l_bounds, u_bounds):
    sample = qmc.scale(sample, l_bounds, u_bounds)
    for i, s in enumerate(l_true_bounds):
        if isinstance(s, int):
            sample[:, i] = np.rint(sample[:, i])
    
    #sample[:, 0] = np.sqrt(sample[:, 0])
    sample = np.abs(sample)
    return sample


sampler = qmc.Sobol(d=dimension, scramble=True, seed=seed)
sample = sampler.random_base2(m=points)
print("Sobol None:", qmc.discrepancy(sample))
sample = my_scale(sample, l_true_bounds, l_bounds, u_bounds)
dists = scipy.spatial.distance.cdist( sample, sample )
np.fill_diagonal(dists, dists[1, 0])
print("Sobol dist:", np.min(dists),  np.max(np.min(dists, axis=0)))
print("Sobol counts:", np.unique(sample[:, 3], return_counts=True))
print("Sobol counts:", np.unique(sample[:, 4], return_counts=True))
print("Sobol counts:", np.unique(sample[:, 5], return_counts=True))

[print(np.min(sample[:, i]), np.max(sample[:, i])) for i in range(6)]



TEMPLATE_run_file = "run_slurm_tf_hw2d_GPU2.14_TensorRT_04032024_TEMPLATE.sh"
tmp_run_file = "tmp_run_slurm_tf_hw2d_GPU2.14_TensorRT_04032024_TEMPLATE.sh"


#remove any duplicate!! Maybe the integers collapsed two samples to the same point
sample = np.unique(sample, axis=0)


for i, values in enumerate(sample):
    subprocess.run(["cp", TEMPLATE_run_file, tmp_run_file]) 
    print(values)
    for j, names in enumerate(template_names):
        if isinstance(l_true_bounds[j], int):
            replacing = "s/" + names + "/" + str(int(values[j])) + "/g"
        else:
            replacing = "s/" + names + "/" + f"{values[j]:.5f}" + "/g"
        print(replacing)
        subprocess.run(["sed", "-i", replacing, tmp_run_file])
    subprocess.run(["sbatch", tmp_run_file])
    #break




