#!/bin/bash

# $1: base folder for training simulations. Make the folders
#     with more structures.
# $2: training folder

sbatch <<EOT
#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=64G
#SBATCH --job-name=ard_$1_emu_$2
#SBATCH -p intel
#SBATCH --output="ard_$1_emu_$2.out"

date

echo "----"
# run python script
python -c "from examples.make_validations import *;\
do_validations(\
folder='$1/$2',\
n_optimization_restarts=20,\
n_fidelities=2,\
turn_off_bias_nargp=False,\
ARD_last_fidelity=True,\
output_folder='output_$1/ard_$2')"

hostname

exit

EOT
