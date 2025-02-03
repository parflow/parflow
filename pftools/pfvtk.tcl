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

# viz

#
# Procedure pfStructuredPoints - Convert a pftools databox to a VTK
#                                StructuredPoints object for display
#
# Parameters - dataset   - The dataset to convert
#
# Return value - The StructuredPoints object

proc Parflow::pfStructuredPoints dataset {

    #
    # Check the arguments
    #
    if {[llength $dataset] != 1} {
	
	set grid  "\nError: Wrong number of arguments\nUsage: pfprintdata dataset\n"
	set code 1
	
    } else {
	
	set code [catch {eval pfgetgrid $dataset} grid]
    }
    
    # Make sure the command completed without error
    if {$code == 1} {
	return  -code error $grid
    }

    if { [info commands vol] == "vol" } {
    } {
	#
	# Create the VTK object
	#
	vtkStructuredPoints vol
	vtkFloatScalars scalars
	[vol GetPointData] SetScalars scalars
    }
    
    # Obtain the grid dimensions
    set gridsize [lindex $grid 0]
    set nx [lindex $gridsize 0]
    set ny [lindex $gridsize 1]
    set nz [lindex $gridsize 2]

    #
    #  Set the geometry information
    #
    vol SetDimensions $nx $ny $nz
    vol SetOrigin 0 0 0
    vol SetSpacing 1.0 1.0 1.0

    #
    # Data is stored in a scalar object
    #
    scalars SetNumberOfScalars [expr $nx * $ny * $nz]
    
    #
    # Copy the data to the scalar object
    #
    # Note: This is slow and we should really write this as a 
    #       C++ routine.
    #
    for {set k 0} {$k < $nz} {incr k} {
	set kOffset [expr $k * $nx * $ny]	
	for {set j 0} {$j < $ny} {incr j} {
	    set jOffset [expr $j * $nx]	    
	    for {set i 0} {$i < $nx} {incr i} {
		set offset [expr $i + $jOffset + $kOffset]		
		scalars SetScalar $offset [pfgetelt $dataset $i $j $k]
	    }
	}
    }
    
    scalars Modified
    #
    # Compute the Min/Max for later use in pipeline
    #
    scalars ComputeRange

    return vol
}
