import numpy as np
import xarray as xr
import yaml

from parflowio.pyParflowio import PFData
from xarray.backends  import BackendEntrypoint

class ParflowBackendEntrypoint(BackendEntrypoint):

    def open_dataset(
        self,
        filename_or_obj,
        *,
        drop_variables=None,
        name='parflow_variable',
        meta_yaml=None,
    ):
        pfd = PFData(filename_or_obj)
        stat = pfd.loadHeader()
        assert stat == 0
        stat = pfd.loadData()
        assert stat == 0
        arr = pfd.viewDataArray()
        dims = list(pfd.getIndexOrder())
        coords = self.decode_coords(pfd)
        da = xr.DataArray(arr, coords=coords, dims=dims, name=name)
        #TODO : Set name and other stuff as necessary (maybe coordinate transfers?
        if meta_yaml:
            meta = self.process_meta(da, yaml)
        return da.to_dataset()

    def load_meta(self, path: str):
        with open(path, 'r') as f:
            meta = yaml.load(f)
        raise NotImplementedError('')

    def decode_coords(self, pfd: PFData):
        """
        Decodes coordinates.

        Parameters
        ----------
        pfd: PFData
            The Parflow Data object

        Returns
        -------
        coords: dict
            Coorinates to pass to the xarray
            datastructure constructor
        """
        x_start, y_start, z_start = pfd.getX(), pfd.getY(), pfd.getZ()
        nx, ny, nz = pfd.getNX(), pfd.getNY(), pfd.getNZ()
        dx, dy, dz = pfd.getDX(), pfd.getDY(), pfd.getDZ()
        coords = {'x': dx * np.arange(0, nx) + x_start,
                  'y': dy * np.arange(0, ny) + y_start,
                  'z': dz * np.arange(0, nz) + z_start, }
        return coords


@xr.register_dataset_accessor("parflow")
class ParflowAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def to_pfb(self):
        raise NotImplementedError('')
