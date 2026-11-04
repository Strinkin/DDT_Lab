nohup python -u train.py \
  --task Height-Flat-Andymini \
  --num_envs 4096 \
  --max_iterations 20000 \
  --device cuda:1 \
  --livestream 2 \
  --kit_args="--/app/livestream/port=49109 --/app/livestream/fixedHostPort=48007" \
  > train.log 2>&1 &
echo $! > train.pid


