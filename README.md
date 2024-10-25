# On Mesa-Optimization in Autoregressively Trained Transformers: Emergence and Capability

This is the official implementation for NeurIPS 2024 paper [On Mesa-Optimization in Autoregressively Trained Transformers: Emergence and Capability](https://arxiv.org/abs/2405.16845).

## Dependencies

```bash
conda env create -f environment.yaml
```

## Hyperparameters Configuration

Detailed hyperparameters config can be found in Appendix B.

## Simulation Experiments

```bash
bash main_train_ar.sh #with hyperparameters in Appendix B
```

## Visualization

```bash
python plot.py #specify the output
```