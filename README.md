## 基于等变学习的图像对称检测

### 环境

```
pytorch==1.7.0
torchvision==0.8.1
cudatoolkit=11.0
matplotlib
albumentations==0.5.2
shapely
opencv-python
tqdm
e2cnn
mmcv
```

### 数据集和训练模型

- download DENDI [onedrive](https://postechackr-my.sharepoint.com/:u:/g/personal/lastborn94_postech_ac_kr/ES2ftVVmTc5Du78EBgfTGy8BwygV_HRa5nWciYeq3cTvoQ?e=y9ETja) or [DENDI](https://github.com/ahyunSeo/DENDI)
- trained [weights](https://postechackr-my.sharepoint.com/:u:/g/personal/lastborn94_postech_ac_kr/EbHHT8lIPThPhYcjU2dLbucBT6jfcNDilC7UXjlSDGKXtA?e=FxdyYk): EquiSym(ours), EquiSym(CNN ver.), pre-trained ReResNet50(D8)

```
.
├── sym_datasets
│   └── DENDI
│       ├── symmetry
│       ├── symmetry_polygon
│       ├── reflection_split.pt
│       ├── rotation_split.pt
│       └── joint_split.pt
├── weights
│   ├── v_equiv_aux_ref_best_checkpoint.pt
│   ├── v_equiv_aux_rot_best_checkpoint.pt
│   ├── v_cnn_ref_best_checkpoint.pt
│   ├── v_cnn_rot_best_checkpoint.pt
│   └── re_resnet50_custom_d8_batch_512.pth
├── (...) 
└── main.py
```

### 可视化样例和测试

- visualize results using the input images in ./imgs

```
    python demo.py --ver equiv_aux_ref -rot 0 -eq --get_theta 10
```

- test with pretrained weights

```
    python train.py --ver equiv_aux_ref -t -rot 0 -eq -wf --get_theta 10
```

- vis(test) with pretrained weights of vanilla CNN model

```
    python demo.py --ver cnn_ref -rot 0
```

### 训练

The trained weights and arguments will be save to the checkpoint path corresponding to the VERSION_NAME.

```
    python train.py --ver VERSION_NAME_REF -tlw 0.01 --get_theta 10 -rot 0 -eq
```

### 可视化

```
具体详情查看README.pdf
```

### 引用

- ReDet [link](https://github.com/csuhan/ReDet)
- e2cnn [link](https://github.com/QUVA-Lab/e2cnn)

