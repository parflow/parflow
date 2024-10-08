# Import the ParFlow TCL package
#
from parflow import Run

Dist_Forcings = Run("Dist_Forcings", __file__)


# set tcl_precision 16

# -----------------------------------------------------------------------------
# File input version number
# -----------------------------------------------------------------------------
Dist_Forcings.FileVersion = 4

# -----------------------------------------------------------------------------
# Process Topology
# -----------------------------------------------------------------------------

Dist_Forcings.Process.Topology.P = 1
Dist_Forcings.Process.Topology.Q = 1
Dist_Forcings.Process.Topology.R = 1

# -----------------------------------------------------------------------------
# Computational Grid
# -----------------------------------------------------------------------------
Dist_Forcings.ComputationalGrid.Lower.X = 0.0
Dist_Forcings.ComputationalGrid.Lower.Y = 0.0
Dist_Forcings.ComputationalGrid.Lower.Z = 0.0

Dist_Forcings.ComputationalGrid.DX = 1000.0
Dist_Forcings.ComputationalGrid.DY = 1000.0
Dist_Forcings.ComputationalGrid.DZ = 20.0

Dist_Forcings.ComputationalGrid.NX = 41
Dist_Forcings.ComputationalGrid.NY = 41
Dist_Forcings.ComputationalGrid.NZ = 24

name = "NLDAS"
# var = ["APCP" "DLWR" "DSWR" "Press" "SPFH" "Temp" "UGRD" "VGRD"]

end = 4
#   for {set i 0} {$i <= $end} {incr i} {
#      foreach v $var {
#       set t1 [expr $i * 24 + 1]
#       set t2 [ expr $t1 + 23]
#       puts $t1
#       puts $t2
#      set filename [format "../NLDAS/%s.%s.%06d_to_%06d.pfb" $name $v $t1 $t2]
#      puts $filename
#     pfdist $filename
#      #pfundist $filename
# }
# }
Dist_Forcings.run()
