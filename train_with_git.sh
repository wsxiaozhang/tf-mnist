#!/bin/bash

echo "Cloning training code from " $GIT_URL
git clone $GIT_URL
echo "Clone code done."

echo "Run training code as: $@"

exec "$@"

echo "Run training done."
