## Demystifying the Neural Tangent Kernel from a Practical Perspective: Can it be trusted for Neural Architecture Search without training? (CVPR 2022)

Jisoo Mok<sup>1</sup>*, Byunggook Na<sup>1</sup>, Ji-Hoon Kim<sup>2,3</sup>†, [Dongyoon Han](https://dongyoonhan.github.io/)<sup>2</sup>†, Sungroh Yoon<sup>1</sup>† <br>
<sub> (†corresponding authors, *works done while at NAVER AI Lab) <br>
 <sup>1</sup>Seoul National University, <sup>2</sup>NAVER AI Lab,  <sup>3</sup>NAVER CLOVA <br><br>

This is the official PyTorch codebase for [Demystifying the Neural Tangent Kernel from a Practical Perspective: Can it be trusted for Neural Architecture Search without training](https://arxiv.org/abs/2203.14577).

### Prepare Image Datasets

- Download CIFAR-10 and CIFAR-100 datasets (available from Torchvision) into ./image_data folder
- Download ImageNet16-120 dataset into ./image_data folder

### Prepare NAS Benchmarks
- Download NAS-Bench-101 API into ./bench_data folder
(https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord)
- Download NAS-Bench-201 API into ./bench_data folder
(https://github.com/D-X-Y/NAS-Bench-201 -> Download NAS-Bench-201-v1_0-e61699.pth)

### Rank correlation at Initialization:
```
cd ntk_nas_init && ./ntk_prediction.sh
```

### Rank correlation Post-training:
```
cd ntk_nas_train && ./ntk_prediction_train.sh
```

### Kernel correlation / Relative Kernel Difference:
```
cd kernel_track && ./nb201_track_ntk.sh
```

### How to cite
```
@inproceedings{mok2022demystifying,
  title={Demystifying the neural tangent kernel from a practical perspective: Can it be trusted for neural architecture search without training?},
  author={Mok, Jisoo and Na, Byunggook and Kim, Ji-Hoon and Han, Dongyoon and Yoon, Sungroh},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11861--11870},
  year={2022}
}
```

### License
```
demystifying-ntk
Copyright (c) 2024-present NAVER Cloud Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

