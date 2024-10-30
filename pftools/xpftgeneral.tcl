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

# Procedure GrabFocus - This procedure is called in order to give a
#           window the focus and keep it until the user performs a
#           specific action such as pressing a `Cancel' or `OK' button.
#
# Parameters - w - The name of the window that will keep the focus
#
# Variables - global done - set to true when the focus is to be released
#
# Return value - None

proc XParflow::GrabFocus {w} {

   set XParflow::done false

   update idletasks
   grab $w
   tkwait variable done
   grab release $w
   
}


# Procedure ErrorDialog - This procedure creates an error dialog
#                         box.
#
# Parameters - msg - The error message to be displayed
#
# Return value - None

proc XParflow::ErrorDialog msg {

   bell

   tk_dialog .error {Error Message} $msg error 0 {OK} 

}
