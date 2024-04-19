CUDA_VISIBLE_DEVICES='0' python nasbench2_ntk_track.py --num_data_workers 4 --train_epoch 100 --sampling 1 --dataset cifar10 --arch_loc ../bench_data/nb201_cf10_arch.json --save_string cf10_kernel_track
CUDA_VISIBLE_DEVICES='0' python nasbench2_ntk_track.py --num_data_workers 4 --train_epoch 100 --sampling 1 --dataset cifar100 --arch_loc ../bench_data/nb201_cf100_arch.json --save_string cf100_kernel_track
CUDA_VISIBLE_DEVICES='0' python nasbench2_ntk_track.py --num_data_workers 4 --train_epoch 100 --sampling 1 --dataset ImageNet16-120 --arch_loc ../bench_data/nb201_imgnet_arch.json --save_string imgnet_kernel_track
