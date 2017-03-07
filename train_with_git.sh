#!/bin/bash

default_ckpt_path='/tmp/train-log/checkpoint'

echo "Cloning training code from " $GIT_URL
git clone $GIT_URL
echo "Done clone code."

echo "Run training code as: " $@

eval "$@"

echo "Done running training code."

ckpt_local_path=$CKPT_LOCAL_PATH
ckpt_remote_path=$CKPT_OSS_PATH

echo 'Persist checkpoints from' $ckpt_local_path 'to ' $ckpt_remote_path

cp -r $ckpt_local_path $ckpt_remote_path

ls -l $ckpt_remote_path

echo "Done persisting checkpoints to remote storage."
