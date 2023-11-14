# vintage-models

Hey lucky people, you have probably lost yourself to end up
here.

Anyway, welcome to my vintage-models repository. Here you will find some personal implementations
of few (I do not yet how many) of the most famous neural network models allowing to process images.

These implementations are made from my personal understanding of the scientific papers introducing
these models. Thus, some implementation/architecture choices can be (are probably) wrong. Even though
the likelihood one of you will really use code from here, please accept my deep apologies for that.

Beside better understanding these models behavior, I am also using this repository
as a road to better learn how to use PyTorch. Thus, in addition of questionable model architecture,
these implementations are probably weak version of "more official" ones.


## How to run the code

### Install Python

```bash
# Install pyenv and pyenv-virtualenv
curl https://pyenv.run | bash

# Configure bash
# see https://github.com/pyenv/pyenv/issues/1906
# tested and working on yueh
echo >> ~/.bashrc 'export PATH="$HOME/.pyenv/bin:$PATH"'
echo >> ~/.bashrc 'eval "$(pyenv init --path)"'
echo >> ~/.bashrc 'eval "$(pyenv init -)"'
echo >> ~/.bashrc 'eval "$(pyenv virtualenv-init -)"'

# Reload bash's configuration
source ~/.bashrc

# Prepare your Ubuntu for building Python
sudo apt install build-essential libz-dev libssl-dev libbz2-dev libreadline-dev libsqlite3-dev libffi-dev

# Install Python
pyenv install 3.10.9
```

### Install the requirements

```bash
pyenv virtualenv 3.10.9 vintage-models
pyenv activate vintage-models
pip install -r requirements.txt
```

### Activate Pre-commits validation

Then activate the `pre-commit` hooks:

```
pre-commit install --allow-missing-config
```

> Warning: if you have aliases for `python` or `pip`, it will break pyenv. Please
verify that you are running the version of Python controlled by pyenv by
running `which python`. If the output does not contain the string `pyenv`,
something is wrong with your install
