# Lottery Ticket Hypothesis
PyTorch implementation of the [Lottery Ticket Hypothesis][lottery].
This implementation uses PyTorch's `prune` module.

## Pre-requisities
Developed in Python `3.7`,
but other versions should work.

If using conda,
run
```bash
conda env create -f environment.yml
```
to create an environment,
`lottery-ticket`,
with all required packages.

## To Run
`main.py` trains a `LeNet`-esque classifier on MNIST
with several rounds of pruning.

```bash
usage: Finding Lottery Tickets on an MNIST classifier [-h] [--lr LR] [--bs BS]
                                                      [--epochs EPOCHS]
                                                      [--prune_pc PRUNE_PC]
                                                      [--prune_rounds PRUNE_ROUNDS]

optional arguments:
  -h, --help            show this help message and exit
  --lr LR               Learning rate (default 1e-3)
  --bs BS               Batch size (default 128)
  --epochs EPOCHS       Number of epochs (default 8)
  --prune_pc PRUNE_PC   Percentage of parameters to prune over the course of
                        the training process (default 0.2)
  --prune_rounds PRUNE_ROUNDS
                        Number of rounds of pruning to perform (default 5)

```

## Contributions
Contributions are welcome.
If opening a PR,
ensure the code conforms to `black` formatting
and `isort` import configurations.

Feel free to open an issue
to ask a question,
raise a bug,
or request new features.

## Project Organization

```
├── README.md          <- The top-level README for developers using this project.
│
├── environment.yml    <- The conda environment file for creating the analysis environment, e.g.
│                         `conda env create -f environment.yml`.
│
├── main.py           <- The training script.
│
├── .gitignore         <- git-ignore configuration file.
│
├── data               <- Directory in which downloaded data will be stored. No data is provided in the repo.
│
├── src                <- source code. Things imported into `main.py`.
```

## TODO
- [X] Implement basic pruning
- [ ] Recreate experiments from the paper
- [ ] Use tensorboard to visualise model and pruning progress

## License
See the full [license](./LICENSE)


[lottery]: https://arxiv.org/abs/1803.03635
