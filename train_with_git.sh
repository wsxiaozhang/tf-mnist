#!/bin/bash

#default_ckpt_path='/output/train-log/checkpoint'
default_output_path='/output'
default_oss_volume_path=$DEFAULT_REMOTE_VOLUME_PATH
default_input_path='/input'

echo "Cloning training code from " $GIT_URL
git clone $GIT_URL
echo "Done clone code."

if [ -d "$default_oss_volume_path" ]; then
  ln -s $default_oss_volume_path $default_input_path
fi

if [ ! -d "$default_output_path" ]; then
  mkdir $default_output_path
fi

echo "Run training code as: " $@

eval "$@"

echo "Done running training code."

ckpt_local_path=$default_output_path
ckpt_remote_path=$DEFAULT_REMOTE_VOLUME_PATH
#ckpt_remote_path=$default_input_path

echo 'Persist checkpoints from' $ckpt_local_path 'to ' $ckpt_remote_path

cp -r $ckpt_local_path'/' $ckpt_remote_path

ls -l $ckpt_remote_path

echo "Done persisting checkpoints to remote storage."
