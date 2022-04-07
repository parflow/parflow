from parflow.tools.builders import TableToProperties


class WellsPropertiesBuilder(TableToProperties):

    def __init__(self, run=None):
        super().__init__(run)

    @property
    def reference_file(self):
        return '/data/parflow/pftools/python/parflow/tools/ref/well_keys.yaml'

    @property
    def key_root(self):
        return self.run.Wells

    @property
    def unit_string(self):
        return 'Wells'

    @property
    def default_db(self):
        return 'conus_1'

    @property
    def db_prefix(self):
        return 'wells_'
