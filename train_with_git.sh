#!/bin/bash

#default_ckpt_path='/output/train-log/checkpoint'
default_output_path='/output'
default_oss_basePath='/var/oss/'
default_input_path='/input'

echo "Cloning training code from " $GIT_URL
git clone $GIT_URL
echo "Done clone code."

mkdir $default_output_path

echo "Run training code as: " $@

eval "$@"

echo "Done running training code."

ckpt_local_path=$default_output_path
#ckpt_remote_path=$default_oss_path + $REMOTE_VOLUME_PATH
ckpt_remote_path=$default_input_path

echo 'Persist checkpoints from' $ckpt_local_path 'to ' $ckpt_remote_path

cp -r $ckpt_local_path $ckpt_remote_path

ls -l $ckpt_remote_path

echo "Done persisting checkpoints to remote storage."
