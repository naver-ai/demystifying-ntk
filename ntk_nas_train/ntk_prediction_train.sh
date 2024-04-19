CUDA_VISIBLE_DEVICES='0' python nasbench2_pred.py --train_epoch 1 --choice others_all_bn --dataset cifar10 --save_string nb201_cf10
CUDA_VISIBLE_DEVICES='0' python nasbench2_pred.py --train_epoch 1 --choice others_all_bn --dataset cifar100 --save_string nb201_cf100
CUDA_VISIBLE_DEVICES='0' python nasbench2_pred.py --train_epoch 1 --val_batch_size 512 --choice others_all_bn --dataset ImageNet16-120 --save_string nb201_imgnet
CUDA_VISIBLE_DEVICES='0' python nasbench2_pred.py --train_epoch 1 --choice others_all_eval_bn --dataset cifar10 --save_string nb201_cf10_eval
CUDA_VISIBLE_DEVICES='0' python nasbench2_pred.py --train_epoch 1 --choice others_all_eval_bn --dataset cifar100 --save_string nb201_cf100_eval
CUDA_VISIBLE_DEVICES='0' python nasbench2_pred.py --train_epoch 1 --val_batch_size 512 --choice others_all_eval_bn --dataset ImageNet16-120 --save_string nb201_imgnet_eval