# LHCb FlavorTagging Trainer

Simplified retraining of with [xgboost](https://github.com/dmlc/xgboost), based
on [tata-antares/tagging_LHCb](https://github.com/tata-antares/tagging_LHCb).

Running `./test.py` will currently train a xgboost model with a given dataset
and check for overtraining via cross-validation afterwards.
All properties of the training are so far taken from the
[`old-tagging.ipynb`](https://github.com/tata-antares/tagging_LHCb/blob/master/old-tagging.ipynb),
from aboves repository.

The trained model will be stored within the `models/`-directory and the
input-data will be stored in a binary xgboost-format in `build/data/`.

After [building](#build-c), run `./build/tag` to load the data previously trained
model within c++.

## Dependencies & Building

### Python

First, [ROOT**5**](https://root.cern.ch/) is needed (there seem to be some
conflicts with ROOT 6).

With ROOT available, install all python-dependencies (into a local [virtual
environment](https://virtualenv.readthedocs.org/en/latest/) for development)
via

```
$ pip install -r requirements.txt
```

### <a name="build-c"></a>c++

For the c++-part you need to compile xgboost yourself, which is well explained
in the [projects
documentation](http://xgboost.readthedocs.org/en/latest/build.html#build-the-shared-library).
After compilation, store the xgboost-project inside a environment-variable
so make can find the libraries and include directories.
```
$ export XGB_REPO=/path/to/xgboost
```

Finally build the executable.
```
$ make
```
