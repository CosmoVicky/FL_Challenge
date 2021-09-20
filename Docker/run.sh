#!/bin/bash
export CUDA_VISIBLE_DEVICES=-1
TF_CONFIG='{"cluster":{"worker":["localhost:12345","localhost:23456","localhost:34567"]},"task":{"type":"worker","index":0}}' python3 ./main.py &> job_0.log &
TF_CONFIG='{"cluster":{"worker":["localhost:12345","localhost:23456","localhost:34567"]},"task":{"type":"worker","index":1}}' python3 ./main.py &> job_1.log &
TF_CONFIG='{"cluster":{"worker":["localhost:12345","localhost:23456","localhost:34567"]},"task":{"type":"worker","index":2}}' python3 ./main.py
python3 ./test.py
