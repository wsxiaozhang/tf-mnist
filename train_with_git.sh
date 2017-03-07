#!/bin/bash

default_ckpt_path='/tmp/train-log/checkpoint'

echo "Cloning training code from ", $GIT_URL
git clone $GIT_URL
echo "Clone code done."

echo "Run training code as: ", "$@"

eval "$@"

echo "Run training done."

ckpt_local_path=$CKPT_LOCAL_PATH
ckpt_remote_path=$CKPT_OSS_PATH

echo 'Saving checkpoints from' $ckpt_local_path 'to ' $ckpt_remote_path

cp -r $ckpt_local_path $ckpt_remote_path

ls -l $ckpt_remote_path

echo "Save checkpoints to remote storage done."
