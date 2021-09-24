import xarray as xr
from xarray.backends  import BackendEntrypoint
from parflowio.pyParflowio imoprt PFData
import numpy as np

class ParflowBackendEntrypoint(BackendEntrypoint):

    def open_dataarray(
        filename_or_obj,
        *,
        drop_variables=None,
        decode_times=True,
        decode_timedelta=True,
        decode_coords=True,
        my_backend_option=None,
    ):
        """
        This currently does nothing
        """
        pfd = PFData(filename_or_obj)
        stat = pfd.loadHeader()
        assert stat == 0
        stat = pfd.loadData()
        assert stat == 0
        arr = pfd.viewDataArray()
        dims = pfd.getIndexOrder().split()
        coords = self.decode_coords(pfd)
        #TODO : Set name and other stuff as necessary (maybe coordinate transfers?
        return xr.DataArray(arr, coords=coords, dim=dims)

    def decode_coords(self, pfd: PFData):
        x_start, y_start, z_start = pfd.getX(), pfd.getY(), pfd.getZ()
        nx, ny, nz = pfd.getNX(), pfd.getNY(), pfd.getNZ()
        dx, dy, dz = pfd.getDX(), pfd.getDY(), pfd.getDZ()
        coords = {
                'x': dx * np.arange(0, nx) + x_start,
                'y': dy * np.arange(0, ny) + y_start,
                'z': dz * np.arange(0, nz) + z_start,
                }
        return coords


@xr.register_dataset_accessor("parflow")
class ParflowAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def to_pfb(self):
        pass
