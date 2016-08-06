To run the notebook on `bam`, first activate the correct miniconda environment.
Zsh-users need to cd into the following directory to setup the root-version.
```shell
cd /opt/rh/miniconda/envs/mkl_test
```
and after that activate the environment via
```shell
source /opt/rh/miniconda/scripts/mkl_test.sh
```
You are now using the `mkl_test` environment. Locally install all (latest)
dependencies:
```shell
pip install --user --upgrade sklearn numpy pandas xgboost==0.4a30 tqdm uncertainties
pip install --user --upgrade root_numpy root_pandas
```
