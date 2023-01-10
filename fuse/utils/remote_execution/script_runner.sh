#usage
source ~/.bashrc
echo "param 1 (gpu num)=":$1
echo "param 2 (conda env name)=":$2
echo "param 3 (':' separated paths to add to PYTHONPATH)=":$3
#all of the next args will be redirected to python
echo "all of the next params will be passed to python: ${@:4}"
unset DISPLAY
source $(dirname $(dirname $(which conda)))/etc/profile.d/conda.sh
conda activate $2

CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$3:$PYTHONPATH python "${@:4}"
#PYTHONPATH=$2:$PYTHONPATH python -u -m ipdb "${@:4}"
#PYTHONPATH=$2:$PYTHONPATH python -u  "${@:4}"
#PYTHONPATH=$2:$PYTHONPATH python -u --version
