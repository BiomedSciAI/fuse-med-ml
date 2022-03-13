#usage
source ~/.bashrc
source $(dirname $(dirname $(which conda)))/etc/profile.d/conda.sh
echo "param 1 (gpu num)=":$1
echo "param 2 (conda env name)=":$2
echo "param 3 (':' separated paths to add to PYTHONPATH)=":$3
echo "param 4 (redirect stdout/stderr to this file)=":$4
#all of the next args will be redirected to python
echo "all of the next params will be passed to python: ${@:5}"
unset DISPLAY
source $(dirname $(dirname $(which conda)))/etc/profile.d/conda.sh
conda activate $2

CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$3:$PYTHONPATH nohup python -u "${@:5}" > $4 2>&1   &    ###ORIG
#PYTHONPATH=$2:$PYTHONPATH python -u -m ipdb "${@:4}" 
#PYTHONPATH=$2:$PYTHONPATH python -u  "${@:4}" 
#PYTHONPATH=$2:$PYTHONPATH python -u --version

