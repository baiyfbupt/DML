# Deep Mutual Learning
Implementation of [Deep Mutual Learning](https://arxiv.org/abs/1706.00384), based on [PaddlePaddle](https://www.paddlepaddle.org.cn/) dygraph mode.

## Usage

```bash
python main
```


## Experiments

| Dataset | Net1 | Net2 |  Independent | DML |
| ------ | ------ | ------ | ------ | ------ |
| CIFAR100 | MobileNet | MobileNet | 73.65/73.65 | 76.32/76.32 |
| CIFAR100 | ResNet50 | ResNet50 | 76.52/76.52 | 76.86/76.86 |
| CIFAR100 | MobileNet | ResNet50 | 73.65/76.52 | 74.16/76.85 |
