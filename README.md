# Deep Mutual Learning
Implementation of [Deep Mutual Learning](https://arxiv.org/abs/1706.00384), based on [PaddlePaddle](https://www.paddlepaddle.org.cn/) dygraph mode.

## Usage

```bash
python main
```

multi-cards training:
```bash
python -m paddle.distributed.launch --selected_gpus=0,1,2,3  --log_dir mylog main.py --use_data_parallel=True
```

## Experiments

| Dataset | Net1 | Net2 |  Independent | DML |
| ------ | ------ | ------ | ------ | ------ |
| CIFAR100 | MobileNet | MobileNet | 73.65/73.65 | 76.32/76.32 |
