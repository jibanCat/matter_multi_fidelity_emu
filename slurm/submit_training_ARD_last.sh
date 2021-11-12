#!/bin/bash

# $1: training folder

sbatch <<EOT
#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=64G
#SBATCH --job-name=hydro_emu_ard_$1
#SBATCH -p intel,batch
#SBATCH --output="hydro_emu_ard_$1.out"

date

cd /rhome/jibancat/bigdata/emu_optimizer

echo "----"
# run python script
~/.conda/envs/simrun/bin/python -c "from examples.make_validations import *;\
do_validations(\
folder='data/$1',\
n_optimization_restarts=20,\
n_fidelities=2,\
turn_off_bias_nargp=False,\
ARD_last_fidelity=True,\
output_folder='output/ard_$1')"

hostname

exit

EOT
