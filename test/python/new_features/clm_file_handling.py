from parflow import Run
from parflow.tools.builders import DomainBuilder
from parflow.tools.export import CLMExporter
from parflow.tools.io import read_clm
import sys

clm = Run("clm", __file__)

#---------------------------------------------------------
# Testing clm data key setting
#---------------------------------------------------------

DomainBuilder(clm) \
    .clm_drv_file(StartDate='2020-01-01', StartTime='00-00-00',
                  StopDate='2020-12-31', StopTime='23-59-59',
                  metf1d='narr_1hr.sc3.txt', outf1d='washita.output.txt',
                  poutf1d='test.out.dat', rstf='washita.rst.')

#---------------------------------------------------------
# Testing clm data readers
#---------------------------------------------------------

clmin_data = read_clm('../../tcl/clm/drv_clmin.dat', type='clmin')
print(clmin_data)

vegm_data = read_clm('../../tcl/clm/drv_vegm.dat', type='vegm')
if not vegm_data[1, 1, 14] == 1:
    sys.exit(1)

vegp_data = read_clm('../../tcl/clm/drv_vegp.dat', type='vegp')
print(vegp_data)

#---------------------------------------------------------
# Testing clm data writers
#---------------------------------------------------------

CLMExporter(clm) \
    .export_drv_clmin() \
    .export_drv_vegm(vegm_data) \
    .export_drv_vegp(vegp_data)




