# pf-xarray
Open parflow files with xarray

`pf-xarray` is an extension to `xarray` that is meant to
make working with Parflow input and output data much easier.

This package introduces extensions which aim to make it simple
to load in such datasets into xarray data structures in a standard
and robust way.

This library is stull under rapid development so APIs may change and
features will be added at a reasonable pace. Check back often to stay
in sync!

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

## Using the `ParflowBinaryReader` and helper functions
The `xarray` backend uses the underlying `ParflowBinaryReader` class to read
the Parflow binary files. It can be used in standalone mode if you do not wish
to interact with files as xarray objects, but rather as numpy arrays. To use
reader you should use it like other [context manager](https://book.pythontips.com/en/latest/context_managers.html)
objects. To open a single file and read the entire dataset you can do:

```
from pf_xarray import ParflowBinaryReader, read_pfb, read_stack_of_pfbs

with ParflowBinaryReader(your_pfb_file) as pfb:
    data = pfb.read_all_subgrids()
```

Similarly, you can select out subsets of the data with the `read_subarray` method.
For instance, to read a `(100, 100, 1)` slice of the data you would use:
```
start_x = 100
start_y = 100
start_z = 0

nx, ny, nz = 100, 100, 1
with ParflowBinaryReader(your_pfb_file) as pfb:
    data = pfb.read_subarray(start_x, start_y, start_z, nx, ny, nz)
```

To facilitate ease of use we also have implemented two helper functions, `read_pfb`
and `read_stack_of_pfbs`. To simply load a full single `pfb` you can do:

```
data = read_pfb(your_pfb_file)
```

In the case that you want to load up a timeseries of pfb files that all have the
same shape you can use the `read_stack_of_pfbs` to efficiently read them into a
single array of shape `(n_timesteps, x, y, z)`. The function does this be reducing
overhead of reading all of the subgrid headers and simply reuses the offsets to read
subgrid data across all of the files. It can be used as:

```
data = read_stack_of_pfbs(your_list_of_pfb_files)
```

There is also support for adding indexing to the `read_stack_of_pfbs` so you don't have
to read the entire spatial extent of all of the files. This is accomplished by adding
a `key` dictionary to the function call. The format of `key` should be:

```
{'x': {'start': start_x, 'stop': end_x},
 'y': {'start': start_y, 'stop': end_y},
 'z': {'start': start_z, 'stop': end_z}}
```

For example, to do something similar to the `ParflowBinaryReader.read_subarray` example:

```
key = {'x': {'start': 100, 'stop': 200},
       'y': {'start': 100, 'stop': 200},
       'z': {'start': 0, 'stop': 1}}
data = read_stack_of_pfbs(your_list_of_pfb_files, key)
```

## Future goals
In the future we hope that this package can provide broader functionality and
make use of the `PFTools` ecosystem. Some planned features are:

 - Compatibility with the `pftools.hydrology` module so that we can calculate
   derived variables seamlessly and efficiently.
 - Ability to be constructed from a `pftools.Run` object.
 - Ability to write out `pfb` files.

Check back for more development notes!
