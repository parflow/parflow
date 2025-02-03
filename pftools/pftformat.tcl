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

#printdata -- This procedure is used to print the elements
#               of a data set as formatted text.
#
#               The data set to be displayed is given as
#               the argument.
#
#               Cmd. Syntax: printdata dataset

proc Parflow::pfprintdata args {

   if {[llength $args] != 1} {

      set grid  "\nError: Wrong number of arguments\nUsage: pfprintdata dataset\n"
      set code 1

   } else {

      set code [catch {eval pfgetgrid $args} grid]

   }

   # Make sure the command completed without error

   if {$code == 1} {
      return  -code error $grid
   }

   # Obtain the grid's dimensions

   set gridsize [lindex $grid 0]
   set nx [lindex $gridsize 0]
   set ny [lindex $gridsize 1]
   set nz [lindex $gridsize 2]

   # Using the dimensions, access each grid element
   # and format the output.

   for {set k 0} {$k < $nz} {incr k} {

      for {set j 0} {$j < $ny} {incr j} {

         for {set i 0} {$i < $nx} {incr i} {

            puts [format "The value at (%3d, %3d, %3d) is: %e" $i $j $k \
                  [pfgetelt $args $i $j $k] ]  

         }

      }

   }

   return
}


#printdiff -- This procedure prints out the differences between
#             corresponding values from two given data sets.
#
#             Criteria must be given to determine which differences
#             are to be returned. The names of two data sets must
#             be given, followed by the number of significant
#             digits to be in each difference, the absolute zero
#             value(optional), and last a file option followed by
#             a file name.
#
# Cmd. Syntax: printdiffs datasetp datasetq sd [abs0]

proc Parflow::pfprintdiff {datasetp datasetq sd {abs0 0}} {

   set dimensions [lindex [pfgetgrid $datasetp] 0]
   set nx [lindex $dimensions 0]
   set ny [lindex $dimensions 1]
   set nz [lindex $dimensions 2]
 
   for {set k 0} {$k < $nz} {incr k} {

      for {set j 0} {$j < $ny} {incr j} {

         for {set i 0} {$i < $nx} {incr i} {

            set difference [pfdiffelt $datasetp $datasetq $i $j $k $sd $abs0]

            if {$difference != ""} {

               puts [format "Absolute difference at (% 3d, % 3d, % 3d): %13e"\
                            $i $j $k $difference]

            }

         }

      }

   }

   set mdiff [pfmdiff $datasetp $datasetq $sd $abs0]
   set mSigDigs [lindex $mdiff 0]
   set maxAbsDiff [lindex $mdiff 1]

   if {$mdiff != ""} {

      puts [format "Minimum significant digits at (% 3d, % 3d, % 3d) = %2d"\
                   [lindex $mSigDigs 0] [lindex $mSigDigs 1]               \
                   [lindex $mSigDigs 2] [lindex $mSigDigs 3]]

      puts [format "Maximum absolute difference = %e" $maxAbsDiff]

   }

}



#printlist -- This procedure lists all of the data sets
#             that have been loaded in as well as their
#             descriptions.
#
#             If no argument is given, all of the data
#             sets and their labels will be listed.
#             If the name of a data set is given then it
#             it and its label will be displayed.
#
#             Cmd. Syntax: printlist [dataset] 

proc Parflow::pfprintlist args {

   # Obtain the list of data sets and labels

   set code [catch {eval pfgetlist $args } datalist]
   
   # Make sure the command was executed properly

   if {$code == 1} {
      return -code error $datalist
   }

   # Print the list elements if any data was loaded

   if {[llength $datalist]} {
      puts "\n\nData set\tLabel"
      puts "--------\t-----"

      foreach i $datalist {
         puts [format "%-10s\t%-s" [lindex $i 0] [lindex $i 1]]
      }    

      puts \n
 
   # No data sets are in memory

   } else {
      puts "\nNo data sets are in memory.\n"
   }

   return 
}
	

#printelt -- This procedure displays the value at a given
#            element.
#
#            It accepts i, j, and k coordinates and the
#            name of a data set.
#
#            Cmd. Syntax: printelt i j k dataset

proc Parflow::pfprintelt args {

   # Obtain the element 

   set code [catch {eval pfgetelt $args} element]
    
   # Make sure the command executed without error

   if {$code == 1} {
      return -code error $element
   }

   # Print the coordinate and element

   puts [format "\nThe value of element (%3d, %3d, %3d) is: %e\n" $i $j $k \
                                                                  $element]

   return
}


#printgrid -- This procedure displays a description of a
#             grid associated with a given data set.  The
#             grid dimensions are printed first, the location
#             of the origin second, and the intervals between
#             each coordinate third.
#
#             The name of a dataset is given as the argument.
#
#             Cmd. Syntax: printgrid dataset

proc Parflow::pfprintgrid args {

   # Obtain the grid information

   set code [catch {eval pfgetgrid $args} grid]

   if {$code == 1} {
      return -code error $grid
   }

   #Print the description of the grid requested

   set dimension [lindex $grid 0]
   set origin    [lindex $grid 1]
   set interval  [lindex $grid 2]

   set nx [lindex $dimension 0]
   set ny [lindex $dimension 1]
   set nz [lindex $dimension 2]

   set x [lindex $origin 0]
   set y [lindex $origin 1]
   set z [lindex $origin 2]
   
   set dx [lindex $interval 0]
   set dy [lindex $interval 1]
   set dz [lindex $interval 2]
 
   puts ""
   puts [format "Grid Dimensions     : (% d, %d, %d)" $nx $ny $nz]
   puts [format "Grid Origin         : (% e, % e, % e)" $x $y $z]
   puts [format "Coordinate Intervals: (% e, % e, % e)" $dx $dy $dz]
   puts ""

   return
}


#printstats -- This command prints the min, max, mean, sum
#              variance, and standard deviation associated
#              with a given data set.
#
#              The name of the data set is given as the argument
#
#              Cmd. Syntax: printstats dataset

proc Parflow::pfprintstats args {

   # Obtain the statistical information

   set code [catch {eval pfgetstats $args} stats]

   if {$code == 1} {
      return -code error $stats
   }

   # Print The statistics

   puts ""
   puts [format "Minimum value: % e  Maximum value: % e" [lindex $stats 0] \
                                                         [lindex $stats 1]]
   puts [format "Mean         : % e  Sum          : % e" [lindex $stats 2] \
                                                         [lindex $stats 3]]
   puts [format "Variance     : % e  Standard Dev.: % e" [lindex $stats 4] \
                                                         [lindex $stats 5]]
   puts ""

   return
}


#printmdiff -- This procedure prints the coordinate at which there is
#              a minimum number of significant digits in the difference
#              between two like points.  The Maximum absolute difference
#              is also printed.

proc Parflow::pfprintmdiff args {

   # Obtain the minimum significant digits and
   # Maximum absolute difference 

   set code [catch {eval pfmdiff $args} diffs]

   # Make sure no error occurred in pfmdiff

   if {$code == 1} {
      return -code error $diffs
   }
    
   set mSigDigs [lindex $diffs 0]

   set maxAbsDiff [lindex $diffs 1]

   puts [format "Minimum significant digits at (%3d, %3d, %3d) = %2d" \
                [lindex $mSigDigs 0] [lindex $mSigDigs 1]           \
                [lindex $mSigDigs 2] [lindex $mSigDigs 3]]

   puts [format "Maximum absolute difference = %e\n" $maxAbsDiff]

   return
}


# Procedure - pfhelp - This procedure is used to give a list of all the
#             PFTools commands and how to use them.
#
# Parameters - 
#
# Variables -
#
# Return value - None

proc Parflow::pfhelp {{cmd all}} {

   puts ""

   switch $cmd {

    pfaxpy        {

                    puts "Usage      : pfaxpy alpha x y\n"
	            puts "Description: This command computes y = alpha*x+y where alpha is a scalar"
                    puts "             and x and y are identifiers representing data sets.  The"
                    puts "             y identifier is returned upon successful completion."

                  }

    pfcvel        {

	            puts "Usage      : pfcvel conductivity phead\n"
	            puts "Description: This command computes the Darcy velocity in cells for the"
	            puts "             conductivity data set represented by the identifier"
                    puts "             `conductivity' and the pressure head data set represented by"
                    puts "             the identifier `phead'.  The identifier of the data set"
                    puts "             created by this operation is returned upon successful completion."
                    puts "             (note: This \"cell\" is not the same as the grid cells; its"
                    puts "             corners are defined by the grid vertices.)"

                  }

    pfdelete      {

                    puts "Usage      : pfdelete dataset\n"
                    puts "Description: This command deletes the data set represented by the"
                    puts "             identifier `dataset'."

                  }

    pfdiffelt     {
               
                    puts "Usage      : pfdiffelt datasetp datasetq i j k sig_digs \[abs_zero\]\n"
	            puts "Description: This command returns the difference of two corresponding"
	            puts "             coordinates from `datasetp' and `datasetq' if the number of"
	            puts "             digits in agreement (significant digits) differs by more"
                    puts "             than `sig_digs' significant digits and the difference is"
                    puts "             greater than the absolute zero given by `abs_zero'."
	         
                  }

    pfenlargebox  {
               
                    puts "Usage      : pfenlargebox dataset new_sx new_sy new_sz\n"
	            puts "Description: This command returns a new dataset which is enlarged"
	            puts "             to be of the new size indicated by sx, sy and sz"
	            puts "             Expansion is done first in z plane, then y plane, and"
	            puts "             x plane."
                  }

    pfflux        {

                    puts "Usage      : pfflux conductivity hhead\n"
	            puts "Description: This command computes the net Darcy flux at vertices for the"
                    puts "             conductivity data set represented by `conductivity' and the"
                    puts "             hydraulic head data set represented by `hhead'.  An identifier"
                    puts "             representing the flux computed will be returned upon successful"
                    puts "             completion."

                  }

    pfgetelt      {

                    puts "Usage      : pfgetelt i j k dataset\n"
	            puts "Description: This command returns the value at element (i,j,k) of data"
                    puts "             set `dataset'.  The i, j, and k above must range from 0 to"
                    puts "             (nx - 1), 0 to (ny - 1), and 0 to (nz - 1) respectively."
                    puts "             The values nx, ny, and nz are the number of grid points"
                    puts "             along the x, y, and z axes respectively.  The string"
                    puts "             `dataset' is an identifier representing the data set whose"
                    puts "             element is to be retrieved."

                  }

    pfgetgrid     {

                    puts "Usage      : pfgetgrid dataset\n"
	            puts "Description: This command returns a description of the grid which serves as"
                    puts "             the domain for data set `dataset'.  The format of the description"
                    puts "             is given below.\n"
	  	    puts "             (nx, ny, nz) - The number of coordinates in each direction"
	  	    puts "             (x ,  y,  z) - The first origin of the grid"
	  	    puts "             (dx, dy, dz) - The distance between each coordinate in each"
                    puts "                            direction\n"
	            puts "             The above information is returned in the following Tcl list"
                    puts "             format:\n"
	            puts "                   {nx ny nz} {x y z} {dx dy dz}"

                  }

    pfgridtype    {

                    puts "Usage      : pfgridtype gridtype\n"
	            puts "Description: This command sets the grid type to either cell centered if"
                    puts "             `gridtype' is set to `cell' or vertex centered if `gridtype'"
                    puts "             is set to `vertex'.  If no new value for `gridtype'"
                    puts "             is given, then the current value of `gridtype' is returned"
                    puts "             The value of `gridtype' will be returned upon successful"
                    puts "             completion of the command." 

                  }

    pfhhead       {

                    puts "Usage      : pfhhead phead\n"
	            puts "Description: This command computes the hydraulic head from the pressure"
                    puts "             head represented by the identifier `phead'.  An identifier for"
                    puts "             the hydraulic head computed is returned upon successful"
                    puts "             completion."

                  }

    pflistdata    {

                    puts "Usage      : pflistdata \[dataset\]\n"
                    puts "Description: This command returns a list of pairs if no argument is given."
                    puts "             The first item in each pair will be the identifier representing"
                    puts "             the data set and the second item will be that data set's label."
                    puts "             If a data set's identifier is given as an argument, then just that"
                    puts "             data set's name and label will be returned."

                  }

    pfload      {

                    puts "Usage      : pfload \[-filetype\] filename\n"
                    puts "Description: This command is used to load data sets that are in"
                    puts "             ParFlow format. If no file type option is given, then"
                    puts "             the extension of the filename is used to determine"
                    puts "             the default file type. An identifier used to represent"
                    puts "             the data set will be returned upon successful completion.\n"
                    puts "             Valid file types are:"
                    puts "                   -pfb  (ParFlow binary)"
                    puts "                   -pfsb (ParFlow scattered binary)"
                    puts "                   -sa   (ParFlow simple ASCII)"
                    puts "                   -sb   (ParFlow simple binary)"
                    puts "                   -rsa  (Real scattered ASCII)\n" 
	            puts "                   -fld  (AVS Field)\n" 
                    puts "             data set will be returned upon successful completion."

                  }

    pfload      {

                    puts "Usage      : pfreload  dataset\n"
                    puts "Description: This command is used to reload a dataset
from disk\n"
}

    pfloadsds     {

                    puts "Usage      : pfloadsds filename dsnum\n"
                    puts "Description: This command is used to load Scientific Data Sets from"
                    puts "             HDF files.  The SDS number `dsnum' will be used to find"
                    puts "             the SDS you wish to load from the HDF file `filename'."
                    puts "             The data set loaded into memory will be assigned an"
                    puts "             identifier which will be used to refer to the data set"
                    puts "             until it is deleted.  This identifier will be returned"
                    puts "             upon successful completion of the command."

                  }

    pfmdiff       {
    
                    puts "Usage      : pfmdiff datasetp datasetq sig_digs \[abs_zero\]\n"
                    puts "Description: If `sig_digs' is greater than or equal to zero, then this"
                    puts "             command computes the grid point at which the number of"
                    puts "             digits in agreement (significant digits) is fewest and"
                    puts "             differs by more than `sig_digs' significant digits.  If"
                    puts "             `sig_digs' is less than zero, then the point at which the"
                    puts "             number of digits in agreement (significant digits) is"
                    puts "             minimum is computed.  Finally, the maximum absolute"
                    puts "             difference is saved computed.  The above information is"
                    puts "             returned in a Tcl list of the following form:\n"
                    puts "             {mi mj mk sd} max_adiff\n"
	            puts "             Given the search criteria, (mi, mj, mk) is the coordinate"
                    puts "             where the minimum number of significant digits `sd' was"
                    puts "             found and `max_adiff' is the maximum absolute difference."

                  }

    pfnewgrid     {

                    puts "Usage      : pfnewdata {nx ny nz} {x y z} {dx dy dz} label\n"
                    puts "Description: This command creates a new data set whose dimension is described"
                    puts "             by the lists {nx ny nz}, {x y z}, and {dx dy dz}.  The first"
                    puts "             list, describes the dimensions, the second indicates the origin,"
                    puts "             and the third gives the length intervals between each coordinate"
                    puts "             along each axis.  The `label' argument will be the label of the"
                    puts "             data set that gets created.  This new data set that is created"
                    puts "             will have all of its data points set to zero automatically.  An"
                    puts "             identifier for the new data set will be returned upon successful"
                    puts "             completion."
                  }

    pfnewlabel    {

                    puts "Usage      : pfnewlabel dataset newlabel\n"
                    puts "Description: This command changes the label of the data set whose identifier is"
                    puts "             `dataset' to `newlabel'."

                  }

    pfphead       {

                    puts "Usage      : pfphead hhead\n"
	            puts "Description: This command computes the pressure head from the hydraulic" 
	            puts "             head represented by the identifier `hhead'.  An identifier for"
                    puts "             the pressure head is returned upon successful completion."

                  }

    pfsavediff    {

                    puts "Usage      : pfsavediff datasetp datasetq sig_digs \[abs_zero\] -file filename"
	            puts "Description: This command saves to a file the differences between the values"
	            puts "             of the data sets represented by `datasetp' and `datasetq' to file"
	            puts "             `filename'.  The data points whose values differ in more than"
                    puts "             `sig_digs' significant digits and whose differences are greater than"
                    puts "             `abs_zero' will be saved.  Also, given the above criteria, the"
	            puts "             minimum number of digits in agreement (significant digits) will be" 
                    puts "             saved."
	  
	            puts "             If `sig_digs' is less than zero, then only the minimum number of"
	            puts "             significant digits and the coordinate where the minimum was"
	            puts "             computed will be saved."
	  
	            puts "             In each of the above cases, the maximum absolute difference given"
	            puts "             the criteria will also be saved."

                  }

    pfsave      {

                    puts "Usage      : pfsave dataset -filetype filename\n"
                    puts "Description: This command is used to save the data set given by the"
                    puts "             identifier `dataset' to a file `filename' of type `filetype'"
                    puts "             in one of the following ParFlow formats:\n"
	            puts "                   pfb  - ParFlow binary format."
	  	    puts "                   sa   - ParFlow simple ASCII format."
	  	    puts "                   sb   - ParFlow simple binary format."
	            puts "                   fld  - AVS Field file format."

                  }

    pfsavesds     {

                    puts "Usage      : pfsavesds dataset -filetype filename\n"
                    puts "Description: This command is used to save the data set given by the"
                    puts "             identifier `dataset' to the file `filename' in the format"
                    puts "             given by `filetype'. The possible HDF formats are float32"
                    puts "             int8, uint8, int16, uint16, int32, and uint32."
          	
                  }

    pfgetstats    {

                    puts "Usage      : pfgetstats dataset\n"
	            puts "Description: This command prints various statistics for the data set"
                    puts "             represented by the identifier `dataset'.  The minimum,"
                    puts "             maximum, mean, sum, variance, and standard deviation are all"
                    puts "             computed.  The above values are returned in a list of the" 
                    puts "             following form:\n"
                    puts "                   {min max mean sum variance (standard deviation)}"

                  }

    pfsubbox      { puts "Usage      : pfgetsubbox dataset il jl kl iu ju ku\n"
	            puts "Description: This command creates a new dataset with the subbox"
                    puts "             starting at il, jl, kl and going to iu, ju, ku."
	          }
 
    pfvmag        {
                    puts "Usage      : pfvmag datasetx datasety datasetz\n"
	            puts "Description: This command computes the velocity magnitude when given three"
                    puts "             velocity components.  The three parameters are identifiers"
                    puts "             which represent the x, y, and z components respectively.  The"
                    puts "             identifier of the data set created by this operation is"
                    puts "             returned upon successful completion."
 
                  }

    pfvvel        {

                    puts "Usage      : pfvvel condition phead\n"
	            puts "Description: This command computes the Darcy velocity in cells for the"
                    puts "             condition data set represented by the identifier `condition'"
                    puts "             and the pressure head represented by the identifier `phead'."
                    puts "             The identifier of the data set created by this operation is"
                    puts "             returned upon successful completion."

                  }

    pfprintdata     {

                    puts "Usage      : pfprintdata dataset\n"
                    puts "Description: This command executes `pfgetgrid' and `pfgetelt' in order to"
                    puts "             display all the elements in the data set represented by the"
                    puts "             identifier `dataset'.\n"
                    puts "             Type `pfhelp printdata' for more information."
        
                  }

    pfprintdiff     {

                    puts "Usage      : pfprintdiff datasetp datasetq sig_digs \[abs_zero\]\n"
	            puts "Description: This command executes the commands `pfdiffelt' and `pfmdiff'"
	            puts "             to print differences to standard output.  The differences are"
                    puts "             printed one per line along with the coordinates where they occur."
                    puts "             The last two lines displayed will show the point at which there"
                    puts "             is a minimum number of significant digits in the difference as"
                    puts "             well as the maximum absolute difference.\n"
                    puts "             Type `pfhelp printdiff' for more information."

                  }

    pfprintgrid   {

                    puts "Usage      : pfprintgrid dataset\n"
	            puts "Description: This command executes pfgetgrid and formats its output"
                    puts "             before printing it on the screen.  The triples (nx, ny, nz),"
                    puts "             (x, y, z), (dx, dy, dz) are all printed on separate lines"
                    puts "             along with labels describing the meaning of each.\n"
                    puts "             Type: `pfhelp pfgetgrid' for more information."

                  }

    pfprintlist     {

                    puts "Usage      : pfprintlist \[dataset\]\n"
                    puts "Description: This command executes pflistdata and formats the output of"
                    puts "             that command.  The formatted output is then printed on the"
                    puts "             screen.  The output consists of a list of data sets and their"
                    puts "             labels one per line if no data set identifier was given or"
                    puts "             just one data set if an identifier was given.\n"
                    puts "             Type `pfhelp printlist' for more information."

                  }

    pfprintmdiff    {

                    puts "Usage      : pfprintmdiff datasetp datasetq sig_digs \[abs_zero\]\n"
                    puts "Description: This command executes `pfmdiff' and formats that command's"
                    puts "             output before displaying it on the screen.  Given the search"
                    puts "             criteria, a line displaying the point at which the difference"
                    puts "             has the least number of significant digits will be displayed."
                    puts "             Another line displaying the maximum absolute difference will"
                    puts "             also be displayed.\n"
                    puts "             Type `ppfhelp printmdiff' for more information."

                  }

   pfprintstats    {

                    puts "Usage      : pfprintstats dataset\n"
                    puts "Description: This command executes `pfgetstats' and formats that command's"
                    puts "             output before printing it on the screen.  Each of the values"
                    puts "             that `pfstats' computes will be displayed along with a label"
                    puts "             describing it.\n"
                    puts "             Type: `pfhelp printstats' for more information."

                  }

   all            {

                    puts "Commands: pfaxpy pfcvel pfdelete pfdiffelt pfflux pfprintelt pfgetgrid pfgridtype"
                    puts "          pfhhead pflistdata pfload pfloadsds pfmdiff pfnewgrid pfnewlabel"
                    puts "          pfphead pfsavediff pfsave pfsavesds pfstats pfvmag pfvvel pfprintdata"
                    puts "          pfprintdiff pfprintgrid pfprintlist printmdiff pfprintstats\n"
                    puts "          Type `pfhelp' followed by a command to obtain detailed descriptions"
                    puts "          of these commands."

                  }

   default        {

                    puts "Unknown command: $cmd"

                  }

   }

   puts ""
     
} 
