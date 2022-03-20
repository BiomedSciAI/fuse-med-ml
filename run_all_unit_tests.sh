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
    PYTHON_VER=3.7
    ENV_NAME="fuse_$PYTHON_VER_"$(sha256sum requirements.txt | awk '{print $1;}')
    echo $ENV_NAME
    mkdir -p ~/env_locks # will create dir if not exist
    lock_filename=~/env_locks/.$ENV_NAME.lock
    echo "Lock filename $lock_filename"

    (
        flock -w 1200 -e 873 || lockfailed # wait for lock at most 20 minutes
        
        # Got lock - excute sensitive code 
        echo "Got lock: $ENV_NAME"   
        
        if find_in_conda_env $ENV_NAME ; then        
            echo "Environment exist: $ENV_NAME"
        else
            # create an environment
            echo "Creating new environment: $ENV_NAME"
            conda create -n $ENV_NAME python=$PYTHON_VER -y
            echo "Creating new environment: $ENV_NAME - Done"

            # install local repository (fuse-med-ml)
            echo "Installing requirements"
            conda run -n $ENV_NAME --no-capture-output --live-stream pip install -e .
            echo "Installing requirements - Done"
        fi
    ) 873>$lock_filename

    # set env name
    ENV_NAME=$ENV_NAME
}


# create environment and run all unit tests
create_env
echo "Running unittests"
conda run -n $ENV_NAME --no-capture-output --live-stream python ./run_all_unit_tests.py
echo "Running unittests - Done"

