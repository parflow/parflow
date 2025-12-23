#BHEADER***********************************************************************
# (c) 1995   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision: 1.4 $
#EHEADER***********************************************************************

#
#  Makefile for parflow tools.
# 
#  This file is machine independent.  Machine dependent information should
#  be in the build script that invokes this makefile
#
PROJECT	= parflow

#
# Project directories -- these may need to be customized for your site
#
# ROOT --	location of the example files.
# TMPDIR --	location for .obj files.
# TOOLS32 --	location of VC++ compiler installation.
# TCL --	location where Tcl is installed.
#

ROOT	= .
TMPDIR	= .
#TOOLS32	= C:\PROGRA~1\DevStudio\VC
TOOLS32	= C:\PROGRA~1\MICROS~2\VC98
TCL	= c:\progra~1\tcl

# comment the following line to compile with symbols
NODEBUG=1

# Tcl version to compile against.

VER	= 83

# Set this to the appropriate value of /MACHINE: for your platform
MACHINE	= IX86

##################
# Project settings
##################

DLLOBJS = pftappinit.obj  printdatabox.obj readdatabox.obj databox.obj \
        error.obj velocity.obj head.obj flux.obj diff.obj stats.obj \
	tools_io.obj axpy.obj getsubbox.obj enlargebox.obj load.obj \
	usergrid.obj grid.obj region.obj file.obj pftools.obj

TOOLS_HDRS = pftools.h printdatabox.h readdatabox.h databox.h error.h \
        velocity.h head.h flux.h diff.h stats.h tools_io.h getsubbox.h \
	enlargebox.h

cc32		= $(TOOLS32)\bin\cl.exe
link32		= $(TOOLS32)\bin\link.exe

CP		= copy
RM		= del
MKDIR		= mkdir

include		= -I$(TOOLS32)\include -I$(TCL)\include
DLL_CFLAGS	= $(cdebug) $(cflags) $(include)
DLL_LIBS	= $(TCL)\lib\tcl$(VER).lib $(conlibsdll) libc.lib

######################################################################
# Link flags
######################################################################

!IFDEF NODEBUG
ldebug = /RELEASE /NODEFAULTLIB
!ELSE
ldebug = /NODEFAULTLIB -debug:full -debugtype:cv
!ENDIF

# declarations common to all linker options
lcommon = /RELEASE /NOLOGO

# declarations for use on Intel i386, i486, and Pentium systems
!IF "$(MACHINE)" == "IX86"
DLLENTRY = @12
lflags   = $(lcommon) -align:0x1000 /MACHINE:$(MACHINE)
!ELSE
lflags   = $(lcommon) /MACHINE:$(MACHINE)
!ENDIF

dlllflags = $(lflags) -entry:_DllMainCRTStartup$(DLLENTRY) -dll

!IF "$(MACHINE)" == "PPC"
libcdll = crtdll.lib
!ELSE
libcdll = msvcrt.lib
!ENDIF

conlibsdll = $(libcdll) kernel32.lib

######################################################################
# Compile flags
######################################################################

!IFDEF NODEBUG
cdebug = -Ox
!ELSE
cdebug = -Z7 -Od -WX
!ENDIF

# declarations common to all compiler options
ccommon = -c -W3 -nologo -DWIN32 -D_WIN32 -D_DLL -DTOOLS_BYTE_SWAP	

!IF "$(MACHINE)" == "IX86"
cflags = $(ccommon) -D_X86_=1
!ELSE
!IF "$(MACHINE)" == "MIPS"
cflags = $(ccommon) -D_MIPS_=1
!ELSE
!IF "$(MACHINE)" == "PPC"
cflags = $(ccommon) -D_PPC_=1
!ELSE
!IF "$(MACHINE)" == "ALPHA"
cflags = $(ccommon) -D_ALPHA_=1
!ENDIF
!ENDIF
!ENDIF
!ENDIF

######################################################################
# Project specific targets
######################################################################

all: $(PROJECT).dll

pkgIndex.tcl: $(PROJECT).dll
	$(TCL)\bin\tclsh$(VER) <<
		pkg_mkIndex $(ROOT) $(PROJECT).dll
<<

$(PROJECT).dll: $(DLLOBJS) $(TOOLS_HDRS)
	$(link32) $(ldebug) $(dlllflags) -out:$@ $(DLL_LIBS) @<<
		$(DLLOBJS)
<<

#######################################################################
# Implicit Targets
#######################################################################


# Implicit Targets

.c.obj:
	$(cc32) $(DLL_CFLAGS) $<

clean:
	-$(RM) $(TMPDIR)\*.obj
	-$(RM) *.dll
	-$(RM) pkgIndex.tcl
	-$(RM) *.lib
	-$(RM) *.exp




