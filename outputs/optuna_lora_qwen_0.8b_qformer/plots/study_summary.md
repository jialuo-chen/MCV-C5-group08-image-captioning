# Optuna Study: lora-optuna-qwen-0.8b-qformer

## Summary

- **Total trials**: 32
- **Completed**: 30
- **Pruned**: 0
- **Failed**: 2
- **Best trial**: #26
- **Best value**: 0.414482

## Best Parameters

- `training.lr`: 0.0007839380084289693
- `tokenizer.max_length`: 128
- `lora.r`: 8
- `lora.alpha`: 2
- `lora.target`: linear_and_head
- `training.optimizer`: adamw
- `training.weight_decay`: 0.0001
- `training.scheduler`: cosine
- `projection.type`: linear
- `projection.num_queries`: 32
- `projection.num_layers`: 1
- `projection.ffn_dim`: 1024

## All Completed Trials

| Trial | Value | training.lr | tokenizer.max_length | lora.r | lora.alpha | lora.target | training.optimizer | training.weight_decay | training.scheduler | projection.type | projection.num_queries | projection.num_layers | projection.ffn_dim |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| 26 | 0.414482 | 0.0007839380084289693 | 128 | 8 | 2 | linear_and_head | adamw | 0.0001 | cosine | linear | 32 | 1 | 1024 |
| 29 | 0.413325 | 0.0009207699104033815 | 256 | 4 | 2 | linear_and_head | adamw | 0.0001 | cosine | linear | 16 | 2 | 1024 |
| 6 | 0.412577 | 5.607308700117546e-05 | 128 | 4 | 2 | linear | adamw | 0.0001 | step | qformer | 32 | 1 | 2048 |
| 12 | 0.412346 | 9.222259227413001e-05 | 128 | 4 | 2 | linear | adamw | 0.001 | step | qformer | 8 | 2 | 2048 |
| 22 | 0.410853 | 0.0002102525264733667 | 128 | 4 | 2 | linear_and_head | adamw | 0.001 | cosine | qformer | 32 | 1 | 1024 |
| 18 | 0.410601 | 7.197226893152498e-05 | 128 | 4 | 2 | linear | adamw | 0.001 | cosine | qformer | 32 | 1 | 1024 |
| 15 | 0.409113 | 0.00027126154083969954 | 256 | 4 | 4 | attention | adamw | 0.001 | step | qformer | 8 | 2 | 2048 |
| 1 | 0.409071 | 5.022563311994765e-05 | 256 | 16 | 2 | all_and_head | adamw | 0.0001 | cosine | qformer | 32 | 2 | 1024 |
| 30 | 0.406098 | 0.00039049962690066353 | 256 | 4 | 4 | linear_and_head | adamw | 0.0001 | cosine | linear | 16 | 2 | 1024 |
| 9 | 0.405427 | 2.6091753957814954e-05 | 128 | 32 | 2 | all_and_head | adamw | 0.0001 | step | qformer | 32 | 1 | 2048 |
| 25 | 0.405382 | 9.703362361179934e-05 | 128 | 32 | 2 | attention | adamw | 0.001 | cosine | qformer | 8 | 2 | 2048 |
| 14 | 0.405195 | 2.5492119179265434e-05 | 128 | 4 | 2 | linear | adamw | 0.001 | step | qformer | 4 | 2 | 2048 |
| 19 | 0.405088 | 4.438308246517662e-05 | 128 | 8 | 4 | linear | adamw | 0.0001 | step | qformer | 32 | 1 | 1024 |
| 11 | 0.405057 | 9.641264700772363e-05 | 256 | 16 | 2 | all_and_head | adamw | 0.0001 | cosine | qformer | 32 | 2 | 1024 |
| 21 | 0.402449 | 3.3417803319227275e-05 | 128 | 4 | 2 | all_and_head | adamw | 0.0001 | step | qformer | 8 | 2 | 1024 |
| 28 | 0.399278 | 0.00185618367683602 | 128 | 8 | 2 | linear_and_head | adamw | 0.0001 | cosine | linear | 32 | 1 | 1024 |
| 23 | 0.395642 | 0.0005735892185463855 | 128 | 4 | 2 | linear_and_head | adamw | 0.01 | cosine | qformer | 4 | 1 | 1024 |
| 8 | 0.394239 | 8.184002205506551e-05 | 128 | 4 | 2 | all | adam | 0.0001 | step | linear | 32 | 1 | 2048 |
| 7 | 0.390391 | 0.0009088474887951092 | 128 | 8 | 4 | all | adam | 0.01 | cosine | linear | 4 | 1 | 2048 |
| 2 | 0.389944 | 1.0296901472345191e-05 | 128 | 8 | 4 | all_and_head | adamw | 0.1 | cosine | linear | 8 | 1 | 2048 |
| 4 | 0.386220 | 4.522999310516458e-05 | 256 | 32 | 2 | linear | adam | 0.01 | step | qformer | 16 | 1 | 1024 |
| 16 | 0.386057 | 0.0008417881949997453 | 128 | 4 | 2 | linear | adam | 0.0001 | step | qformer | 32 | 1 | 1024 |
| 10 | 0.379596 | 0.0008308384824642156 | 128 | 4 | 2 | all | adamw | 0.0001 | step | qformer | 16 | 1 | 2048 |
| 20 | 0.342316 | 2.0338332068990517e-05 | 128 | 4 | 2 | linear | adamw | 0.01 | none | linear | 32 | 1 | 2048 |
| 31 | 0.322379 | 0.0015414094491625172 | 256 | 16 | 2 | linear_and_head | sgd | 0.001 | cosine | linear | 16 | 2 | 1024 |
| 13 | 0.254051 | 0.000354158764260736 | 128 | 4 | 2 | linear | sgd | 0.001 | step | qformer | 8 | 2 | 2048 |
| 0 | 0.101757 | 7.274917088027814e-05 | 128 | 4 | 2 | attention | sgd | 0.001 | step | linear | 16 | 2 | 1024 |
| 3 | 0.093509 | 3.3610226697378736e-05 | 256 | 8 | 2 | linear_and_head | sgd | 0.0001 | step | qformer | 32 | 1 | 1024 |
| 5 | 0.075443 | 6.0926173694724706e-05 | 256 | 4 | 2 | full_attention | sgd | 0.0 | none | linear | 32 | 1 | 1024 |
| 24 | 0.074219 | 2.774999668734031e-05 | 128 | 4 | 2 | linear | sgd | 0.0001 | step | qformer | 8 | 1 | 2048 |