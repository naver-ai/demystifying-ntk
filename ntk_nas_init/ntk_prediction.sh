CUDA_VISIBLE_DEVICES='0' python nasbench1_pred.py --choice others_all_bn --dataset cifar10 --save_string nb101_cf10
CUDA_VISIBLE_DEVICES='0' python nasbench1_pred.py --choice others_all_eval_bn --dataset cifar10 --save_string nb101_cf10_eval

CUDA_VISIBLE_DEVICES='0' python nasbench2_pred.py --choice others_all_bn --dataset cifar10 --save_string nb201_cf10
CUDA_VISIBLE_DEVICES='0' python nasbench2_pred.py --choice others_all_bn --dataset cifar100 --save_string nb201_cf100
CUDA_VISIBLE_DEVICES='0' python nasbench2_pred.py --choice others_all_bn --val_batch_size 512 --dataset ImageNet16-120 --save_string nb201_imgnet
CUDA_VISIBLE_DEVICES='0' python nasbench2_pred.py --choice others_all_eval_bn --dataset cifar10 --save_string nb201_cf10_eval
CUDA_VISIBLE_DEVICES='0' python nasbench2_pred.py --choice others_all_eval_bn --dataset cifar100 --save_string nb201_cf100_eval
CUDA_VISIBLE_DEVICES='0' python nasbench2_pred.py --choice others_all_eval_bn --val_batch_size 512 --dataset ImageNet16-120 --save_string nb201_imgnet_eval