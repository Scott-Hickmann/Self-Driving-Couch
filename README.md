# Self-Driving-Couch

## Note: The repo has been moved to https://github.com/jclin22009/motorized_couch.

## Installation

Create a new conda environment and install the dependencies:

```bash
conda env create -n couch -f environment.yml python=3.10
```

Activate the environment:

```bash
conda activate couch
```

## Development

After each time you install or remove a dependency, make sure to run:

```bash
conda env export > environment.yml
```

To install packages added by others, make sure to run:

```bash
conda env update --file environment.yml --prune
```

## Usage

```bash
python src/YOUR_FILE_NAME.py
```
