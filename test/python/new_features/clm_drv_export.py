from parflow import Run
from parflow.tools.builders import DomainBuilder
from parflow.tools.export import CLMDriverExporter

clm = Run("clm", __file__)

DomainBuilder(clm) \
  .clm_drv_file(StartDate='2020-01-01', StartTime='00-00-00',
                StopDate='2020-12-31', StopTime='23-59-59',
                metf1d='narr_1hr.sc3.txt', outf1d='washita.output.txt',
                poutf1d='test.out.dat', rstf='washita.rst.')


CLMDriverExporter(clm).export_drv_clmin()



