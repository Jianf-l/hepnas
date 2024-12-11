
## DARTS space


### Search
```bash
python train_search_ws_edge_final.py --dataset [cifar10/cifar100] --batch_size 128 --split_ckpts 15,25,35,45 --warmup_epoch 10 --kd_loss multi_teacher --projection_warmup_epoch 5
```

The settings of random order or Reversed order of segmentation is the same as above, you only need to change the py file. This will reported the searched architecture (named genotype) in the terminal. Copy the genotype into genotypes.py and create alias for it.


### Evaluation
```shell
python train.py/train_imagenet.py --arch [alias for the genotype]
```


