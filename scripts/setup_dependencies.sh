#!/bin/bash -eux

cd $(dirname $0)/..

pip install -r requirements_dev.txt -r requirements_opt.txt
