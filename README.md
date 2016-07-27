# LHCb FlavorTagging Trainer

Prototypes for FlavourTagging reoptimization scripts

## Running on lxplus
I strongly recommend setting up an isolated, clean python environment using
Anaconda, therefore download conda (mabye check for a newer version
[here](https://www.continuum.io/downloads))
```bash
wget http://repo.continuum.io/archive/Anaconda3-4.1.1-Linux-x86_64.sh
```
and install it (might take a while)
```
bash Anaconda3-4.1.1-Linux-x86_64.sh
```
You could also look at miniconda and manually install some packages.

Install ROOT for the conda environment
```
conda install root
```

Update/install some python packages
```
pip install --update xgboost pandas root_pandas tqdm
```
