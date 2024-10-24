#BHEADER**********************************************************************
#
#  Copyright (c) 1995-2024, Lawrence Livermore National Security,
#  LLC. Produced at the Lawrence Livermore National Laboratory. Written
#  by the Parflow Team (see the CONTRIBUTORS file)
#  <parflow@lists.llnl.gov> CODE-OCEC-08-103. All rights reserved.
#
#  This file is part of Parflow. For details, see
#  http://www.llnl.gov/casc/parflow
#
#  Please read the COPYRIGHT file or Our Notice and the LICENSE file
#  for the GNU Lesser General Public License.
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License (as published
#  by the Free Software Foundation) version 2.1 dated February 1999.
#
#  This program is distributed in the hope that it will be useful, but
#  WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms
#  and conditions of the GNU General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public
#  License along with this program; if not, write to the Free Software
#  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
#  USA
#**********************************************************************EHEADER

proc pfstrip {input_filename output_filename} {

    set input_file [open $input_filename r]
    set output_file [open $output_filename w]

    while { [gets $input_file line] >= 0} {
	if {![regexp "(^#.*$)|(^\ *$)" $line]} {
	    puts $output_file $line
	}
    }

    close $input_file
    close $output_file
}

set program "parflow"

set runname [lindex $argv 0]

# If user did not specify process group assume none
if {$argc > 1} {
    set processgroup [lindex $argv 1]
} {
    set processgroup ""
}


# Get the number of processor to run on
set file [open .amps.info.$processgroup]
gets $file num_procs
close $file

#pfstrip $runname.in.solver $runname.in.solver.strp
#pfstrip $runname.in.problem $runname.in.problem.strp

exec $env(PARFLOW_DIR)/bin/$program $num_procs $runname

#file delete $runname.in.solver.strp $runname.in.problem.strp
