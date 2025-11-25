# Installation

Our codebase is built upon **Python 3.12, PyTorch 2.4.0 (recommended)**. 

## Setup scripts

```shell
conda create -n MOTIP python=3.12		# suggest to use virtual envs
conda activate MOTIP
# PyTorch:
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
# Other dependencies:
conda install pyyaml tqdm matplotlib scipy pandas
pip install wandb accelerate einops
# Compile the Deformable Attention:
cd models/ops/
sh make.sh
# [Optional] After compiled, you can use following script to test it:
python test.py
```


