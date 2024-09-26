#!/bin/bash -l

# Log files (-o output, -e errors).
#SBATCH -o logs/%j_gpu1_undistr.out.undistr
#SBATCH -e logs/%j_gpu1_undistr.err.undistr

# Working directory.
#SBATCH -D ./

# Job name.
#SBATCH -J training_transformer_on_raw

# Resources.
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=0
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:4

# Email notification settings.
#SBATCH --mail-type=end
#SBATCH --mail-user=kammnoel1@gmail.com

# Wall clock limit.
#SBATCH --time=02:00:00

# Load environment modules.
source /etc/profile.d/modules.sh
module purge  # Clean all currently loaded modules.

# Load other required modules first.
module load anaconda/3/2023.03
module load pytorch/gpu-cuda-11.6/2.0.0
module load scikit-learn/1.2.2
module load huggingface-transformers/4.31.0

# Activate virtual environment.
source ${HOME}/venvs/eeg_llm/bin/activate

# Upgrade pip and install required Python packages.
#pip install --upgrade pip
pip install transformers torch scikit-learn numpy pandas matplotlib

# --- TF related flags (provided by Tim) ---

# Avoid CUPTI warning message
export LD_LIBRARY_PATH=${CUDA_HOME}/extras/CUPTI/lib64:${LD_LIBRARY_PATH}

# Avoid OOM
export TF_FORCE_GPU_ALLOW_GROWTH=true

## XLA
# cuda aware
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME}"
# enable autoclustering for CPU and GPU
export TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"

# Run your Python script.
srun python transformer_on_toy_data_clean.py