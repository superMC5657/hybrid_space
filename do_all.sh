collection=msrvtt10k
space=latent
visual_feature=resnet-152
rootpath=/home1/zhangyan/downloads/VisualSearch
overwrite=1
model=$2
epochs=10

# training
gpu=$3

CUDA_VISIBLE_DEVICES=$gpu python trainer.py --rootpath $rootpath --overwrite $overwrite --max_violation --text_norm --visual_norm \
                                            --collection $collection --visual_feature $visual_feature --space $space --model $model --num_epochs $epochs \
                                            --overwrite 1

# python trainer.py --rootpath /home/supermc/VisualSearch --overwrite 1 --max_violation --text_norm --visual_norm --collection msrvtt10k --visual_feature resnext101-resnet152 --space latent --num_epochs 10 --model dual_encoding_attention