# Evidential-Bayesian-DL

A PyTorch framework for uncertainty aware image classification. It trains and
evaluates three families of models under one configuration system.

- **Deterministic networks.** A softmax network trained with cross entropy, and
an evidential deep learning (EDL) network with a Dirichlet output layer.
- **Bayesian neural networks (BNN).** Weight posteriors sampled with SGMCMC, built on [posteriors](https://github.com/normal-computing/posteriors).
- **Evidential Bayesian neural networks (eBNN).** A Dirichlet output layer
sampled with SGMCMC under a Gamma prior on the total concentration. The model
combines the epistemic uncertainty of a BNN with the distributional
uncertainty of EDL.

The framework supports single label targets and annotator count targets.


## Installation

The project was developed and tested with **Python 3.10.19**.

### 1. Clone the repository

```bash
git clone https://github.com/jpweideman/Evidential-Bayesian-DL.git
cd Evidential-Bayesian-DL
```


### 2. Install Poetry

```bash
curl -sSL https://install.python-poetry.org | python3 -
# or
wget -qO- https://install.python-poetry.org | python3 -
```


### 3. Ensure Poetry is on PATH

If `poetry --version` fails with `command not found`, add Poetry's bin
directory to your shell `PATH`, reload your shell configuration, and run
`poetry --version` again.


### 4. Install Python 3.10 with pyenv

Install pyenv for your OS first, then run:

```bash
pyenv install 3.10.19
pyenv local 3.10.19
poetry env use "$(pyenv which python)"
```


### 5. Install the dependencies

```bash
poetry install
```


### 6. Activate the virtual environment

```bash
source $(poetry env info --path)/bin/activate
```

You can also prefix every command with `poetry run` instead.

## Usage



### Train a model

Every experiment config under `configs/` trains one model with one command.
Hydra overrides change any value from the command line.

```bash
python train.py --config-name fashion_mnist_sgd
python train.py --config-name cifar10_dirichlet_bnn_sgld training.prior_fs.params.rate=0.26
python train.py --config-name cifar10_categorical_bnn_sgld training.sampler.params.temperature=0.01
```

The experiment configs, one per dataset and method:


| Method                 | Fashion-MNIST                        | CIFAR-10                       |
| ---------------------- | ------------------------------------ | ------------------------------ |
| Softmax network (SGD)  | `fashion_mnist_sgd`                  | `cifar10_sgd`                  |
| EDL network            | `fashion_mnist_edl`                  | `cifar10_edl`                  |
| Categorical BNN (SGLD) | `fashion_mnist_categorical_bnn_sgld` | `cifar10_categorical_bnn_sgld` |
| Evidential BNN (SGLD)  | `fashion_mnist_dirichlet_bnn_sgld`   | `cifar10_dirichlet_bnn_sgld`   |


For annotator counts on CIFAR-10H:


| Method                              | CIFAR-10H                                 |
| ----------------------------------- | ----------------------------------------- |
| Multinomial network (SGD)           | `cifar10h_multinomial_sgd`                |
| Dirichlet multinomial network (MAP) | `cifar10h_dirichlet_multinomial_map_sgd`  |
| Multinomial BNN (SGLD)              | `cifar10h_multinomial_bnn_sgld`           |
| Dirichlet multinomial eBNN (SGLD)   | `cifar10h_dirichlet_multinomial_bnn_sgld` |


Each config trains on the train split, selects the checkpoint on the
validation split, and evaluates the test split at the end.

### Warm starts

`training.pretrained` loads the weights of another run before training:

```bash
python train.py --config-name cifar10_categorical_bnn_sgld \
    training.pretrained.enabled=true \
    training.pretrained.path=outputs/my_sgd_run/best_model.pt
```

For a Dirichlet output layer under a `gamma_strength` prior,
`training.pretrained.match_prior_mode=true` adds one constant to the output
bias. The pretrained model then starts with its median total concentration at
the prior mode, and its class predictions do not change.

### Run experiment sets

`run_experiments.py` runs the entries of an experiments yaml in order, once per
seed, into `outputs/<entry>_s<seed>/`. `experiments_e1.yaml` and
`experiments_e2.yaml` are examples.

```bash
python run_experiments.py --file_name experiments_e1.yaml --runs 3
python run_experiments.py --file_name experiments_e1.yaml --list
```

Completed runs are recorded in `.<experiments file>_state.json`, so an
interrupted set continues where it stopped. `--only`, `--skip`, and `--rerun`
take entry names. An entry can declare `pretrained_from` to warm start from
another entry's best checkpoint of the same seed.

### Resume a run

`hydra.run.dir` sets the output directory of a run. Without it, Hydra writes
to `outputs/<date>/<time>/`. `last_checkpoint.pt` is written there every
epoch. To continue an interrupted run, start it again with the same config,
the same overrides, and the same directory:

```bash
python train.py --config-name cifar10_sgd hydra.run.dir=outputs/my_run/
```

If the directory holds a `last_checkpoint.pt`, training restores the weights,
the optimizer state, the random number generator state, the epoch, the
posterior sample list, and the W&B run, and continues to
`training.num_epochs`. `.hydra/config.yaml` in the run directory records the
config and overrides of the original run. Pretrained weights and the bias
shift are applied on a fresh start only.

### Run outputs

Each run writes one directory:

```
outputs/<run>/
├── .hydra/config.yaml     The composed config the run used
├── best_model.pt          Best checkpoint on the checkpoint split
├── last_checkpoint.pt     Written every epoch, for resuming
├── samples/               Posterior samples (sampled runs)
├── arrays/<split>.npz     Per input arrays, for every split with an array_dump metric
├── arrays/summary.json    Final metrics of every evaluation split
└── metrics.json           The Weights and Biases run summary
```



### Logging

Weights and Biases logging is on by default. Metric names are
`<split>/<metric>`. Set the project with `training.wandb.project=<name>`, or
turn logging off with `training.wandb.enabled=false`. The per split metrics
are also written to `arrays/summary.json`, so no W&B account is needed to
read the results.

## Configuration

The framework is config driven. A yaml file describes the whole run, and the
code contains no experiment specific logic. Hydra composes every experiment
config from four groups:

```yaml
defaults:
  - datasets: cifar10             # configs/datasets/    data loaders per split
  - model: resnet20               # configs/model/       architecture and output layer
  - training: standard            # configs/training/    optimizer or sampler, loss, priors, logging
  - evaluation: standard_cifar10  # configs/evaluation/  metrics and interval per split
  - _self_
```

Every component in a config has the same shape, a `name` and optional
`params`:

```yaml
training:
  sampler:
    name: sgld
    params:
      lr: 0.0001
      temperature: 1.0
  prior_fs:
    name: gamma_strength
    params:
      concentration: 27
      rate: 0.26
```

At start up, a builder in `src/builders/` reads each block, looks the `name`
up in the registry in `src/registry.py`, and constructs the component with
`params`. Any value can be changed from the command line without editing a
file, for example `training.sampler.params.lr=0.001`.

A training config defines either `optimizer` or `sampler`, not both. Learning
rate schedulers apply to optimizers only. An evaluation interval of `N` runs
the split every N epochs, `-1` runs it after the final epoch only, and `0`
disables it.

## Extending the framework

There is one registry for each component type: models, output layers, losses,
likelihoods, priors, function space priors, metrics, optimizers, schedulers,
samplers, datasets, and transforms. To add a component you need no change to
the training code. Two steps:

1. Put a module in the matching `src/` folder and register the class under a
  name. The folder imports every module on start up, so the registration
   runs by itself.
2. Name it in a yaml, with its parameters under `params`.
  ```yaml
   training:
     prior_fs:
       name: my_prior
       params:
         scale: 2.0
  ```

The registered names are the single source of truth, and each `src/` folder
holds one module per component, so the folder listing shows what exists.

## Repository structure

```
Evidential-Bayesian-DL/
├── configs/
│   ├── datasets/ model/ training/ evaluation/   Config groups
│   └── *.yaml                                   Experiment configs (dataset x method)
├── src/
│   ├── models/ losses/ likelihoods/ priors/ priors_fs/
│   ├── samplers/ optimizers/ schedulers/ metrics/ data/
│   ├── training/                                Engines, evaluators, handlers, checkpoints
│   ├── builders/                                Config to component, through the registry
│   └── registry.py
├── experiments_*.yaml                           Experiment sets for run_experiments.py
├── train.py
└── run_experiments.py
```



## Acknowledgements

The framework builds on [PyTorch](https://pytorch.org/) and
[PyTorch Ignite](https://pytorch-ignite.ai/) for training,
[posteriors](https://github.com/normal-computing/posteriors) for the SGMCMC
samplers, [edl-pytorch](https://github.com/teddykoker/evidential-learning-pytorch)
for the evidential losses, [torchmetrics](https://lightning.ai/docs/torchmetrics/)
for the calibration metrics, [Hydra](https://hydra.cc/) for configuration,
[Weights and Biases](https://wandb.ai/) for logging, and
[Poetry](https://python-poetry.org/) for dependency management.
