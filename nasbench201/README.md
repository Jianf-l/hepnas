## Experiments on NAS-Bench-201

### Dataset preparation
1. Download the [NAS-Bench-201-v1_0-e61699.pth](https://drive.google.com/file/d/1SKW0Cu0u8-gb18zDpaAGi0f74UdXeGKs/view) and save it under `data` folder.

2. Install NAS-Bench-201 via pip
```
pip install nas-bench-201
```

### Running HEP-NAS on NAS-Bench-201

```bash
python train_search.py --dataset [cifar10/cifar100/imagenet16-120] --edge_crit hep --split_ckpts 10,20,30 --kd_loss multi_teacher --subnets_training_epochs 15 --batch_size 128
```

The performance of searched architecture will be reported directly in the terminal, using api.query_by_arch() function. You can set the batch_size to other numbers according to your device.

#### 