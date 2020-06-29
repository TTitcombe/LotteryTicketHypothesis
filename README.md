# Lottery Ticket Hypothesis
PyTorch implementation of the [Lottery Ticket Hypothesis][lottery].

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
`main.py` trains a `LeNet`-esque classifier on MNIST.
Run `python main.py -h` to see optional command line arguments.

## Contributions
Contributions are welcome.
If opening a PR,
ensure the code conforms to `black` formatting
and `isort` import configurations.

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

## License
See the full [license](./LICENSE)


[lottery]: https://arxiv.org/abs/1803.03635
