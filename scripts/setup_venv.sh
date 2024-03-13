#!/bin/bash -eux

cd $(dirname $0)/..

virtualenv ./venv

. ./venv/bin/activate

./setup_dependencies.sh
