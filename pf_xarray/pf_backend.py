import xarray as xr
from xarray.backends  import BackendEntrypoint

class ParflowBackendEntrypoint(BackendEntrypoint):

    def open_dataset(
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
        pass
