dataset_name="custom"
config_name="bridge_r101.yaml"
gpu=0
CUDA_VISIBLE_DEVICES=$gpu python3 -u test.py \
      --config config/$dataset_name/$config_name
