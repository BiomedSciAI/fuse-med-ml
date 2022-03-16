#!/bin/bash

find_in_conda_env(){
    conda env list | grep "${@}" >/dev/null 2>/dev/null
}
lockfailed()
{
        echo "failed to get lock"
        exit 1
}
create_jenkins_env() {
    PYTHON_VER=3.7
    ENV_NAME="jenkins_fuse_$PYTHON_VER_"$(sha256sum requirements.txt | awk '{print $1;}')
    echo $ENV_NAME
    mkdir -p ~/jenkins_env_locks # will create dir if not exist
    lock_filename=~/jenkins_env_locks/.$ENV_NAME.lock
    echo "Lock filename $lock_filename"
    (
        flock -w 1200 -e 873 || lockfailed # wait for at most 20 minutes   
        if find_in_conda_env $ENV_NAME ; then        
            echo "Environment exist: $ENV_NAME"
        else
            echo "Creating new environment: $ENV_NAME"
            conda create -n $ENV_NAME python=$PYTHON_VER -y
            echo "Creating new environment: $ENV_NAME - Done"
            echo "Activating new environment: $ENV_NAME"
            CONDA_BASE=$(conda info --base)
            source "$CONDA_BASE/etc/profile.d/conda.sh"
            conda activate $ENV_NAME
            echo "Activating new environment: $ENV_NAME - Done"
            echo "Installing requirements"
            pip install -e .
            echo "Installing requirements"
            conda deactivate
        fi
        sleep 120
    ) 873>$lock_filename

    # return value
    JENKINS_ENV_NAME=$ENV_NAME
}

