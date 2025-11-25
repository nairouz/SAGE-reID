# Installation
⬇️ Our codebase is built upon **Python 3.12, PyTorch 2.5.0 (recommended)**. 

## Setup scripts

```shell
conda create -n SAGE-reID python=3.12		# suggest to use virtual envs
conda activate SAGE-reID
# PyTorch:
# CUDA 12.4
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
# Other dependencies:
pip install -t requirments.txt
```


