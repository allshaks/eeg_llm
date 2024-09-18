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
#SBATCH --mem=30000
# Uncomment the next line to use GPU (a100), adjust as needed.
# --gres=gpu:a100:1

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
module load scikit-learn/1.2.2
module load pytorch/cpu/2.1.0
module load huggingface-transformers/4.31.0

# Load GCC module last.
module load gcc/14

# Export compiler variables to use the correct GCC compilers.
export CC=$(which gcc)
export CXX=$(which g++)

# Set LDFLAGS to ensure the correct libstdc++ is used.
export LDFLAGS="$LDFLAGS -L$(gcc --print-file-name=libstdc++.so.6 | sed 's/libstdc++.so.6//') -Wl,-rpath,$(gcc --print-file-name=libstdc++.so.6 | sed 's/libstdc++.so.6//')"

# Activate virtual environment.
source ${HOME}/venvs/eeg_llm/bin/activate

# Upgrade pip and install required Python packages.
pip install --upgrade pip
pip install transformers torch scikit-learn numpy pandas matplotlib

# Run your Python script.
srun python transformer_on_toy_data_clean.py