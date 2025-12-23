#! /usr/bin/perl
#BHEADER**********************************************************************
#
#  Copyright (c) 1995-2024, Lawrence Livermore National Security,
#  LLC. Produced at the Lawrence Livermore National Laboratory. Written
#  by the Parflow Team (see the CONTRIBUTORS file)
#  <parflow\@lists.llnl.gov> CODE-OCEC-08-103. All rights reserved.
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

use strict;

use File::Basename;
use File::Find;
use Cwd;

my $cpytxt = "";

$cpytxt = <<END;
#BHEADER**********************************************************************
#
#  Copyright (c) 1995-2024, Lawrence Livermore National Security,
#  LLC. Produced at the Lawrence Livermore National Laboratory. Written
#  by the Parflow Team (see the CONTRIBUTORS file)
#  <parflow\@lists.llnl.gov> CODE-OCEC-08-103. All rights reserved.
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
END

$cpytxt = substr($cpytxt, 0, -1);

# Flush I/O on write to avoid buffering
$|=1;

my $debug=4;
my $end_of_line = $/;
my @filesToProcess = ();

#
# Remove duplicated values
#
sub unique {
    foreach my $test (@_){
	my $i = -1;
	my @indexes = map {$i++;$_ eq $test ? $i : ()} @_;
	shift @indexes;
	foreach my $index (@indexes){
	    splice(@_,$index,1);
	}
    }
    return @_;
}

my $pwd = cwd;

#=============================================================================
# .h files
#=============================================================================

#
# Find the files to convert.
@filesToProcess = ();
print @filesToProcess if $debug > 2;

#
# Build list of files in which to look for templates.
#
sub selectHFiles {
    print "Thinking about $File::Find::name\n" if $debug > 1;
    if ( $File::Find::name =~ m!/(.svn|CVS|include|scripts|\{arch\})$!o ) {
	$File::Find::prune = 1;
    }
    elsif ( -f && m/(.*\.sh$)|(.*\.tcl$)|(.*mc$)|(run)/o ) {
	push @filesToProcess, $File::Find::name;
    }
}
find( \&selectHFiles, '.' );
for my $file (@filesToProcess) {
    print "Working on $file\n" if $debug > 1;

    my $directory = dirname $file;
    my $filebasename = basename $file;

    #
    # Read in whole file into a single variable
    #
    open FILE, "< $file" || die "Cannot open file $file";
    undef $/;
    my $str = "";
    while (<FILE>) {
	$str .= $_;
    }

    #
    # Do substitutions
    #
    $str =~ s!#BHEADER(.*?)EHEADER\**!$cpytxt!sgm;

    #
    # Write new file
    #
    rename( $file, $file . ".BAK");
    open TEMPFILE, "> $file" || die "Cannot open new file $file";
    print TEMPFILE $str;
    close TEMPFILE || die "Cannot close file $file";
}
