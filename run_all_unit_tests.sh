#!/bin/bash

# check if current env already exist
find_in_conda_env(){
    conda env list | grep "${@}" >/dev/null 2>/dev/null
}

# error message when failed to lock
lockfailed()
{
    echo "failed to get lock"
    exit 1
}

# create an environment including the set of requirements specified in fuse-med-ml/requirements.txt (if not already exist)
create_env() {
    force_cuda_version=$1
    env_path=$2
    mode=$3

    requirements=$(cat requirements.txt)
    if [ $mode = "examples" ]; then
        requirements+=$(cat examples/requirements.txt)
    fi

    PYTHON_VER=3.7
    ENV_NAME="fuse_$PYTHON_VER-$(echo -n $requirements | sha256sum | awk '{print $1;}')"
    echo $ENV_NAME
    
    # env full name
    if [ $env_path = "no" ]; then
        env="-n $ENV_NAME"
    else
        env="-p $env_path/$ENV_NAME"
    fi


    # create a lokc
    mkdir -p ~/env_locks # will create dir if not exist
    lock_filename=~/env_locks/.$ENV_NAME.lock
    echo "Lock filename $lock_filename"

    (
        flock -w 1200 -e 873 || lockfailed # wait for lock at most 20 minutes

        # Got lock - excute sensitive code
        echo "Got lock: $ENV_NAME"

        nvidia-smi

        if find_in_conda_env $ENV_NAME ; then
            echo "Environment exist: $env"
        else
            # create an environment
            echo "Creating new environment: $env"
            conda create $env python=$PYTHON_VER -y
            echo "Creating new environment: $env - Done"

            if [ $force_cuda_version != "no" ]; then
                echo "forcing cudatoolkit $force_cuda_version"
                conda install $env pytorch torchvision cudatoolkit=$force_cuda_version -c pytorch -y
                echo "forcing cudatoolkit $force_cuda_version - Done"
            fi

            # install local repository (fuse-med-ml)
            echo "Installing core requirements"
            conda run $env --no-capture-output --live-stream pip install -r requirements.txt
            echo "Installing core requirements - Done"

            if [ $mode = "examples" ]; then
                echo "Installing examples requirements"
                conda run $env --no-capture-output --live-stream pip install -r examples/requirements.txt
                echo "Installing examples requirements - Done"
            fi
        fi
    ) 873>$lock_filename

    # set env name
    ENV_TO_USE=$env
}


# create environment and run all unit tests
echo "input args ($#) - $@"

if [ "$#" -gt 0 ]; then
    force_cuda_version=$1
else
    force_cuda_version="no"
fi

if [ "$#" -gt 1 ]; then
    env_path=$2
else
    env_path="no"
fi

echo "Create core env"
create_env $force_cuda_version $env_path "core"
echo "Create core env - Done"

echo "Running core unittests in $ENV_TO_USE"
conda run $env --no-capture-output --live-stream python ./run_all_unit_tests.py core
echo "Running core unittests - Done"

echo "Create examples env"
create_env $force_cuda_version $env_path "examples"
PYTHONPATH=$PYTHONPATH:./examples
echo "Create examples env - Done"

echo "Running examples unittests in $ENV_TO_USE"
conda run $env --no-capture-output --live-stream python ./run_all_unit_tests.py examples
echo "Running examples unittests - Done"
