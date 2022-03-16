#!/bin/bash

create_jenkins_env() {
    PYTHON_VER=3.7
    ENV_NAME="jenkins_fuse_$PYTHON_VER_"$(sha256sum requirements.txt | awk '{print $1;}')
    echo $ENV_NAME
    find_in_conda_env(){
        conda env list | grep "${@}" >/dev/null 2>/dev/null
    }
    #CONDA_BASE=$(conda info --base)
    if find_in_conda_env $ENV_NAME ; then
        echo "Environment exist: $ENV_NAME"
    else
        echo "Creating new environment: $ENV_NAME"
        conda create -n $ENV_NAME python=$PYTHON_VER -y
        echo "Creating new environment: $ENV_NAME - Done"
        echo "Activating new environment: $ENV_NAME"
        conda activate $ENV_NAME
        echo "Activating new environment: $ENV_NAME - Done"
        echo "Installing requirements"
        pip install -e .
        echo "Installing requirements"
    fi
    # return value
    JENKINS_ENV_NAME=$ENV_NAME
}

