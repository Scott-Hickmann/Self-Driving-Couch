# Self-Driving-Couch

## Installation

Create a new conda environment and install the dependencies:

```bash
conda env create -n couch -f environment.yml python=3.10
```

Activate the environment:

```bash
conda activate couch
```

Set up libomp:

```bash
brew install libomp
```

## Development

After each time you install or remove a dependency, make sure to run:

```bash
conda env export > environment.yml
```

## Usage

```bash
python src/YOUR_FILE_NAME.py
```