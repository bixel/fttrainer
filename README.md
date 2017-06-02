# LHCb FlavorTagging Trainer

Prototypes for FlavourTagging reoptimization scripts

## Usage
Currently, only the `crossval_training.py` and `xgboost_training.py` are
actively developed. Each script uses configuration data from the `configs/`
directory.

### Cut Selection and Bootstrapped Crossvalidation
Given a set of cut-parameters (defined in a tagger-specific config), the
hyperparameters need to be validated to prevent overtraining.

This can be done with the `crossval_training.py` script, e.g. like
``` bash
./crossval_training.py -c configs/someconfig.json -p roc_curve_plot.pdf
```
which will read the given configuration, print out average mistag power values
and plot the average roc curve, obtained in the bootstrapping step.

To speed up the read-in and selection step, the script is able to either write
the selected tuple to disk (and only printout average tagging power values) via
``` bash
./crossval_training.py -c config.json -o selected_tuple.root
```
or read a preselected file via
``` bash
./crossval_training.py -c config.json -i selected_tuple.root -p plot.pdf
```

### Training XGBoost for production
After the hyperparameters have been verified, a XGBoost model can be trained
with the `xgboost_training.py` script. It will read files obtained in the
previous step (i.e. *selected_tuple.root*), train a XGBoost classifier with the
hyperparameters given in the configuration, calculate the per-event tagging
power and write out the trained model.

``` bash
./xgboost_training.py -c config.json -i selected_tuple.root -o predicted_tuple.root -s xboost_model.model
```

### Calibration
The best choice is to use the
[EPM](https://gitlab.cern.ch/lhcb-ft/EspressoPerformanceMonitor) here.

## Running on lxplus
I strongly recommend setting up an isolated, clean python environment using
Anaconda (miniconda), therefore download conda via
```bash
wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
```
and install it with
```bash
bash Miniconda2-latest-Linux-x86_64.sh
```
this will create a virtual environment with its own python version in
`~/miniconda2`.

To provide a recent version of `gcc` and ROOT which will work with the new
environment run
```bash
SetupProject ROOT 6.06.06
```
and finally install the python dependencies via
```
~/miniconda2/bin/pip install numpy
~/miniconda2/bin/pip install pandas root-numpy root-pandas matplotlib sklearn xgboost tqdm
```
Once, all the dependencies are installed, you can start `jupyter`
```bash
jupyter notebook --port 61337 --no-browser
```
where `--port` is any free port. If the port is already in use, jupyter will
automatically use the next free one.

To access the notebook from your desktop, just forward a local port to the
remote port on lxplus. You need to make sure that you are referencing the same
lxplus instance (e.g. define `Host lxplus042.cern.ch` for your lxplus part of
`~/.ssh/config`).
```bash
ssh -N -f -L "8888:[::1]:61337" lxplus
```
will then forward your local port 8888 to the remote notebook. Just open the
browser and visit [localhost:8888](http://localhost:8888).
