#!/bin/bash

# Set the PYTHONPATH environment variable
export PYTHONPATH=$PYTHONPATH:./Grounded_Segment_Anything/recognize-anything
export PYTHONPATH=$PYTHONPATH:./Grounded_Segment_Anything/GroundingDINO
export PYTHONPATH=$PYTHONPATH:./Grounded_Segment_Anything/segment_anything

# Set CUDA environment variable
export CUDA_HOME=/usr/local/cuda-12.4
alias python=python3

# Set build options
export BUILD_WITH_CUDA=True
export AM_I_DOCKER=False