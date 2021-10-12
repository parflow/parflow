# pf-xarray
Open parflow files with xarray

`pf-xarray` is an extension to `xarray` that is meant to
make working with Parflow input and output data much easier.

This package introduces extensions which aim to make it simple
to load in such datasets into xarray data structures in a standard
and robust way.

## Getting started

This project is in the very early stages, so you will have to install
from source to use `pf-xarray`:

```
git clone https://github.com/arbennett/pf-xarray.git
cd pf-xarray
python setup.py develop
```

## Reading data
Once installed there is nothing to import. This package implements backends
and extensions that are directly hooked into `xarray` itself. You can use
the functionality to read Parflow data with:

```
import xarray as xr

# Read a single 'pfb' file
da = xr.open_dataarray(path_to_pfb, name=varname)

# Read several 'pfb' files given a 'pfmetadata' file
ds = xr.open_dataset(path_to_pfmetadata)
```

## Writing data
Although it is not yet supported, `pf-xarray` will also add a `.parflow` accessor
to `xarray` which will support writing out to 'pfb' files. This will eventually look
something like:

```
output_pfb_files = ds.parflow.to_pfb(output_template_string)
```

Check back for more development notes!
