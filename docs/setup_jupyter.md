# Local Jupyter execution

Notes for running the repository notebooks in a local Jupyter environment.

## Requirements

- Linux or a Jupyter-compatible environment
- Python 3.10 o 3.11
- JupyterLab o Notebook
- NVIDIA GPU with drivers installed, if CUDA acceleration is desired

## Create an environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

## Install PyTorch

Before installing PyTorch, verify your NVIDIA installation:

```bash
nvidia-smi
```

Install PyTorch with CUDA support using the variant recommended by PyTorch for your system. Example:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

For CPU:

```bash
pip install torch torchvision torchaudio
```

## Install auxiliary dependencies

```bash
pip install -r requirements.txt
```

## Start Jupyter
```bash
jupyter lab
```

or
```bash
jupyter notebook
```

## Verify GPU visibility from PyTorch

Inside the notebook:
```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

## Suggested execution order

First, try a short training run:
```python
history_overfit, model_overfit = train_overfitnet(
    num_epochs=3, batch_size=128, normalize=False, plot_curves=True
)
```

Then run the grid search.

## Notes

- MNIST data is downloaded to `./data`.
- Results are saved to `./runs` by default.
- `gTTS` is an optional dependency used only for audio notifications.
