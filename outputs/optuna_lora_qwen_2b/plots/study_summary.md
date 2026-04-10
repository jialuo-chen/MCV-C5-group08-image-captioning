# Optuna Study: lora-optuna-qwen-2b

## Summary

- **Total trials**: 33
- **Completed**: 30
- **Pruned**: 0
- **Failed**: 3
- **Best trial**: #27
- **Best value**: 0.421484

## Best Parameters

- `training.lr`: 4.019237202699583e-05
- `tokenizer.max_length`: 128
- `lora.r`: 4
- `lora.alpha`: 4
- `lora.target`: attention
- `training.optimizer`: adamw
- `training.weight_decay`: 0.1
- `training.scheduler`: step
- `projection.type`: qformer
- `projection.num_queries`: 32
- `projection.num_layers`: 1
- `projection.ffn_dim`: 2048

## All Completed Trials

| Trial | Value | training.lr | tokenizer.max_length | lora.r | lora.alpha | lora.target | training.optimizer | training.weight_decay | training.scheduler | projection.type | projection.num_queries | projection.num_layers | projection.ffn_dim |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| 27 | 0.421484 | 4.019237202699583e-05 | 128 | 4 | 4 | attention | adamw | 0.1 | step | qformer | 32 | 1 | 2048 |
| 25 | 0.420506 | 8.74999864016184e-05 | 128 | 4 | 4 | attention | adamw | 0.1 | step | qformer | 32 | 1 | 2048 |
| 6 | 0.419056 | 5.607308700117546e-05 | 128 | 4 | 2 | linear | adamw | 0.0001 | step | qformer | 32 | 1 | 2048 |
| 12 | 0.416779 | 0.0002024506482963534 | 128 | 4 | 2 | linear | adamw | 0.01 | step | qformer | 32 | 1 | 2048 |
| 29 | 0.416390 | 7.079597826542527e-05 | 128 | 4 | 4 | attention | adamw | 0.001 | step | qformer | 32 | 2 | 2048 |
| 9 | 0.416292 | 2.6091753957814954e-05 | 128 | 32 | 2 | all_and_head | adamw | 0.0001 | step | qformer | 32 | 1 | 2048 |
| 18 | 0.415847 | 0.0001227156831886742 | 128 | 4 | 2 | linear_and_head | adamw | 0.01 | step | qformer | 32 | 1 | 1024 |
| 14 | 0.415272 | 4.5847383219208547e-05 | 128 | 4 | 2 | linear | adamw | 0.001 | step | qformer | 4 | 1 | 2048 |
| 23 | 0.414408 | 2.263543618219561e-05 | 128 | 32 | 4 | all_and_head | adamw | 0.0 | none | qformer | 32 | 2 | 2048 |
| 21 | 0.414255 | 0.00035744619736624995 | 128 | 4 | 2 | linear | adamw | 0.01 | none | linear | 4 | 1 | 1024 |
| 15 | 0.412874 | 0.00017113028536319038 | 128 | 16 | 2 | linear | adamw | 0.1 | step | qformer | 32 | 1 | 2048 |
| 1 | 0.412163 | 5.022563311994765e-05 | 256 | 16 | 2 | all_and_head | adamw | 0.0001 | cosine | qformer | 32 | 2 | 1024 |
| 11 | 0.410877 | 2.1672113149230604e-05 | 128 | 32 | 2 | all_and_head | adamw | 0.0001 | step | qformer | 32 | 1 | 2048 |
| 8 | 0.408463 | 8.184002205506551e-05 | 128 | 4 | 2 | all | adam | 0.0001 | step | linear | 32 | 1 | 2048 |
| 20 | 0.407695 | 0.00011978774355684339 | 256 | 4 | 2 | linear | adamw | 0.0001 | none | qformer | 16 | 1 | 2048 |
| 31 | 0.407498 | 7.109988645300661e-05 | 128 | 8 | 4 | attention | adamw | 0.001 | step | qformer | 4 | 1 | 1024 |
| 2 | 0.401248 | 1.0296901472345191e-05 | 128 | 8 | 4 | all_and_head | adamw | 0.1 | cosine | linear | 8 | 1 | 2048 |
| 13 | 0.391600 | 0.0004490978778060302 | 128 | 32 | 2 | linear | adamw | 0.01 | cosine | qformer | 32 | 1 | 2048 |
| 7 | 0.390127 | 0.0009088474887951092 | 128 | 8 | 4 | all | adam | 0.01 | cosine | linear | 4 | 1 | 2048 |
| 4 | 0.382651 | 4.522999310516458e-05 | 256 | 32 | 2 | linear | adam | 0.01 | step | qformer | 16 | 1 | 1024 |
| 28 | 0.370963 | 4.596033503947338e-05 | 256 | 4 | 4 | attention | adamw | 0.1 | step | linear | 32 | 1 | 2048 |
| 10 | 0.368793 | 0.0008308384824642156 | 128 | 4 | 2 | all | adamw | 0.0001 | step | qformer | 16 | 1 | 2048 |
| 16 | 0.356068 | 0.0015303457048777822 | 128 | 4 | 2 | full_attention | sgd | 0.01 | step | qformer | 32 | 1 | 2048 |
| 32 | 0.285895 | 1.9263920409785048e-05 | 128 | 4 | 4 | attention | adam | 0.1 | step | qformer | 4 | 1 | 2048 |
| 19 | 0.260170 | 0.0004471189586292786 | 128 | 4 | 4 | linear | sgd | 0.1 | step | qformer | 32 | 1 | 1024 |
| 30 | 0.133357 | 3.663772183252697e-05 | 128 | 16 | 4 | all | sgd | 0.1 | step | qformer | 32 | 2 | 2048 |
| 0 | 0.109056 | 7.274917088027814e-05 | 128 | 4 | 2 | attention | sgd | 0.001 | step | linear | 16 | 2 | 1024 |
| 17 | 0.098307 | 1.0580437429106431e-05 | 128 | 4 | 2 | linear | sgd | 0.0001 | cosine | qformer | 32 | 1 | 2048 |
| 3 | 0.083467 | 3.3610226697378736e-05 | 256 | 8 | 2 | linear_and_head | sgd | 0.0001 | step | qformer | 32 | 1 | 1024 |
| 5 | 0.079908 | 6.0926173694724706e-05 | 256 | 4 | 2 | full_attention | sgd | 0.0 | none | linear | 32 | 1 | 1024 |