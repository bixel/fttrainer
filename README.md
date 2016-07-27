# LHCb FlavorTagging Trainer

Prototypes for FlavourTagging reoptimization scripts

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
