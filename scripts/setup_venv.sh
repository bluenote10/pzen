#!/bin/bash -eux

cd $(dirname $0)/..

virtualenv ./venv

. ./venv/bin/activate

pip install -r requirements_dev.txt
