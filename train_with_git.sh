#!/bin/bash

default_output_path='/output'
default_oss_volume_path=$DEFAULT_REMOTE_VOLUME_PATH
default_input_path='/input'

# get user's training code
echo "Cloning training code from " $GIT_URL
git clone $GIT_URL
echo "Done cloning code."

# install user's modules
repo=${GIT_URL##*/}
prj=${repo/%.git/}
if [ -f "./$prj/requirements.txt" ];then
  echo "Found ./$prj/requirements.txt, start to install modules."
  pip install -r ./$prj/requirements.txt
fi

# default input dir
if [ -d "$default_oss_volume_path" ]; then
  ln -s $default_oss_volume_path $default_input_path
fi

# default output dir
if [ ! -d "$default_output_path" ]; then
  mkdir -p $default_output_path
fi

# exec user's command
echo "Run training code as: " $@
eval "$@"
echo "Done running training code."

# auto persist outputs to user's oss volume
ckpt_local_path=$default_output_path
ckpt_remote_path=$default_oss_volume_path
if [ -d $ckpt_remote_path ]; then
  cp -r $ckpt_local_path/* $ckpt_remote_path
  echo "Persists checkpoints from local path $ckpt_local_path/ to remote data volume $ckpt_remote_path/" 
  ls -l $ckpt_remote_path
else
  echo "Cannot find remote data volume $ckpt_remote_path, checkpoints are not persisted remotely."
fi
echo "Done persisting checkpoints to remote storage."
