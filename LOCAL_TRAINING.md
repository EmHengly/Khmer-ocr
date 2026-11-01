# Local training setup

This repository keeps the application and training notebooks in Git. Large
checkpoints, datasets, generated outputs, and uploaded files are intentionally
excluded by `.gitignore`.

## Clone and install

```bash
git clone <repository-url>
cd word_spotting_app
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirement.txt
```

For GPU training, install the PyTorch build that matches the local CUDA
version before installing the remaining requirements.

## Prepare the dataset

The training notebooks currently use Kaggle-specific paths such as
`/kaggle/input/...`. Before running them locally, change those paths to the
local dataset directory and keep the labels CSV and image directory together.
Do not commit the dataset unless it is intentionally public and small enough
for the repository.

## Run training

Open the relevant notebook under `Model/Model/` in VS Code or Jupyter and run
the cells in order. Set the output directory to a local ignored directory,
for example:

```python
output_dir = "checkpoints/local"
```

After training, use the generated checkpoint path in the application. The
application expects the model directory to contain the Hugging Face model and
processor files, including `config.json`, `model.safetensors`, tokenizer files,
and `preprocessor_config.json`.

## Run the application

Update the model path in the selected `app*.py` file to the checkpoint created
locally, then start Flask with the project's normal command.

Never commit model weights, optimizer state, datasets, uploaded documents, or
credentials. Store large checkpoints separately using rsync, object storage, or
another model registry.
