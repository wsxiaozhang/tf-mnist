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
  mkdir -p $default_output_path
fi

echo "Run training code as: " $@

eval "$@"

echo "Done running training code."

ckpt_local_path=$default_output_path
ckpt_remote_path=$default_oss_volume_path

if [ -d $ckpt_remote_path ]; then
  cp -r $ckpt_local_path/* $ckpt_remote_path
  echo Persists checkpoints from local path $ckpt_local_path/ to remote data volume $ckpt_remote_path/ 
  ls -l $ckpt_remote_path
else
  echo Cannot find remote data volume $ckpt_remote_path, checkpoints are not persisted remotely.
fi

echo "Done persisting checkpoints to remote storage."
