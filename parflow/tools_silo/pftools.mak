# Microsoft Developer Studio Generated NMAKE File, Format Version 4.10
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Console Application" 0x0103

!IF "$(CFG)" == ""
CFG=pftools - Win32 Debug
!MESSAGE No configuration specified.  Defaulting to pftools - Win32 Debug.
!ENDIF 

!IF "$(CFG)" != "pftools - Win32 Release" && "$(CFG)" !=\
 "pftools - Win32 Debug"
!MESSAGE Invalid configuration "$(CFG)" specified.
!MESSAGE You can specify a configuration when running NMAKE on this makefile
!MESSAGE by defining the macro CFG on the command line.  For example:
!MESSAGE 
!MESSAGE NMAKE /f "pftools.mak" CFG="pftools - Win32 Debug"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "pftools - Win32 Release" (based on "Win32 (x86) Console Application")
!MESSAGE "pftools - Win32 Debug" (based on "Win32 (x86) Console Application")
!MESSAGE 
!ERROR An invalid configuration is specified.
!ENDIF 

!IF "$(OS)" == "Windows_NT"
NULL=
!ELSE 
NULL=nul
!ENDIF 
################################################################################
# Begin Project
# PROP Target_Last_Scanned "pftools - Win32 Debug"
CPP=cl.exe
RSC=rc.exe

!IF  "$(CFG)" == "pftools - Win32 Release"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "Release"
# PROP BASE Intermediate_Dir "Release"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "Release"
# PROP Intermediate_Dir "Release"
# PROP Target_Dir ""
OUTDIR=.\Release
INTDIR=.\Release

ALL : ".\pftools.exe"

CLEAN : 
	-@erase "$(INTDIR)\axpy.obj"
	-@erase "$(INTDIR)\databox.obj"
	-@erase "$(INTDIR)\diff.obj"
	-@erase "$(INTDIR)\error.obj"
	-@erase "$(INTDIR)\flux.obj"
	-@erase "$(INTDIR)\getsubbox.obj"
	-@erase "$(INTDIR)\grid.obj"
	-@erase "$(INTDIR)\head.obj"
	-@erase "$(INTDIR)\pftappinit.obj"
	-@erase "$(INTDIR)\pftools.obj"
	-@erase "$(INTDIR)\printdatabox.obj"
	-@erase "$(INTDIR)\readdatabox.obj"
	-@erase "$(INTDIR)\region.obj"
	-@erase "$(INTDIR)\stats.obj"
	-@erase "$(INTDIR)\tools_io.obj"
	-@erase "$(INTDIR)\usergrid.obj"
	-@erase "$(INTDIR)\velocity.obj"
	-@erase ".\pftools.exe"

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

# ADD BASE CPP /nologo /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_CONSOLE" /YX /c
# ADD CPP /nologo /MD /W3 /GX /O2 /I "c:\program files\tcl\include" /D "NDEBUG" /D "WIN32" /D "_CONSOLE" /D "TOOLS_BYTE_SWAP" /YX /c
CPP_PROJ=/nologo /MD /W3 /GX /O2 /I "c:\program files\tcl\include" /D "NDEBUG"\
 /D "WIN32" /D "_CONSOLE" /D "TOOLS_BYTE_SWAP" /Fp"$(INTDIR)/pftools.pch" /YX\
 /Fo"$(INTDIR)/" /c 
CPP_OBJS=.\Release/
CPP_SBRS=.\.
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
BSC32_FLAGS=/nologo /o"$(OUTDIR)/pftools.bsc" 
BSC32_SBRS= \
	
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib /nologo /subsystem:console /machine:I386
# ADD LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib tcl76.lib tk42.lib /nologo /subsystem:console /machine:I386 /out:"pftools.exe"
LINK32_FLAGS=kernel32.lib user32.lib gdi32.lib winspool.lib advapi32.lib\
 shell32.lib ole32.lib oleaut32.lib uuid.lib tcl76.lib tk42.lib /nologo\
 /subsystem:console /incremental:no /pdb:"$(OUTDIR)/pftools.pdb" /machine:I386\
 /out:"pftools.exe" 
LINK32_OBJS= \
	"$(INTDIR)\axpy.obj" \
	"$(INTDIR)\databox.obj" \
	"$(INTDIR)\diff.obj" \
	"$(INTDIR)\error.obj" \
	"$(INTDIR)\flux.obj" \
	"$(INTDIR)\getsubbox.obj" \
	"$(INTDIR)\grid.obj" \
	"$(INTDIR)\head.obj" \
	"$(INTDIR)\pftappinit.obj" \
	"$(INTDIR)\pftools.obj" \
	"$(INTDIR)\printdatabox.obj" \
	"$(INTDIR)\readdatabox.obj" \
	"$(INTDIR)\region.obj" \
	"$(INTDIR)\stats.obj" \
	"$(INTDIR)\tools_io.obj" \
	"$(INTDIR)\usergrid.obj" \
	"$(INTDIR)\velocity.obj"

".\pftools.exe" : "$(OUTDIR)" $(DEF_FILE) $(LINK32_OBJS)
    $(LINK32) @<<
  $(LINK32_FLAGS) $(LINK32_OBJS)
<<

!ELSEIF  "$(CFG)" == "pftools - Win32 Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "Debug"
# PROP BASE Intermediate_Dir "Debug"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "Debug"
# PROP Intermediate_Dir "Debug"
# PROP Target_Dir ""
OUTDIR=.\Debug
INTDIR=.\Debug

ALL : ".\pftools.exe" "$(OUTDIR)\pftools.bsc"

CLEAN : 
	-@erase "$(INTDIR)\axpy.obj"
	-@erase "$(INTDIR)\axpy.sbr"
	-@erase "$(INTDIR)\databox.obj"
	-@erase "$(INTDIR)\databox.sbr"
	-@erase "$(INTDIR)\diff.obj"
	-@erase "$(INTDIR)\diff.sbr"
	-@erase "$(INTDIR)\error.obj"
	-@erase "$(INTDIR)\error.sbr"
	-@erase "$(INTDIR)\flux.obj"
	-@erase "$(INTDIR)\flux.sbr"
	-@erase "$(INTDIR)\getsubbox.obj"
	-@erase "$(INTDIR)\getsubbox.sbr"
	-@erase "$(INTDIR)\grid.obj"
	-@erase "$(INTDIR)\grid.sbr"
	-@erase "$(INTDIR)\head.obj"
	-@erase "$(INTDIR)\head.sbr"
	-@erase "$(INTDIR)\pftappinit.obj"
	-@erase "$(INTDIR)\pftappinit.sbr"
	-@erase "$(INTDIR)\pftools.obj"
	-@erase "$(INTDIR)\pftools.sbr"
	-@erase "$(INTDIR)\printdatabox.obj"
	-@erase "$(INTDIR)\printdatabox.sbr"
	-@erase "$(INTDIR)\readdatabox.obj"
	-@erase "$(INTDIR)\readdatabox.sbr"
	-@erase "$(INTDIR)\region.obj"
	-@erase "$(INTDIR)\region.sbr"
	-@erase "$(INTDIR)\stats.obj"
	-@erase "$(INTDIR)\stats.sbr"
	-@erase "$(INTDIR)\tools_io.obj"
	-@erase "$(INTDIR)\tools_io.sbr"
	-@erase "$(INTDIR)\usergrid.obj"
	-@erase "$(INTDIR)\usergrid.sbr"
	-@erase "$(INTDIR)\vc40.idb"
	-@erase "$(INTDIR)\vc40.pdb"
	-@erase "$(INTDIR)\velocity.obj"
	-@erase "$(INTDIR)\velocity.sbr"
	-@erase "$(OUTDIR)\pftools.bsc"
	-@erase "$(OUTDIR)\pftools.pdb"
	-@erase ".\pftools.exe"
	-@erase ".\pftools.ilk"

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

# ADD BASE CPP /nologo /W3 /Gm /GX /Zi /Od /D "WIN32" /D "_DEBUG" /D "_CONSOLE" /YX /c
# ADD CPP /nologo /MD /W3 /Gm /GX /Zi /Od /I "c:\program files\tcl\include" /D "_DEBUG" /D "WIN32" /D "_CONSOLE" /D "TOOLS_BYTE_SWAP" /FR /YX /c
CPP_PROJ=/nologo /MD /W3 /Gm /GX /Zi /Od /I "c:\program files\tcl\include" /D\
 "_DEBUG" /D "WIN32" /D "_CONSOLE" /D "TOOLS_BYTE_SWAP" /FR"$(INTDIR)/"\
 /Fp"$(INTDIR)/pftools.pch" /YX /Fo"$(INTDIR)/" /Fd"$(INTDIR)/" /c 
CPP_OBJS=.\Debug/
CPP_SBRS=.\Debug/
# ADD BASE RSC /l 0x409 /d "_DEBUG"
# ADD RSC /l 0x409 /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
BSC32_FLAGS=/nologo /o"$(OUTDIR)/pftools.bsc" 
BSC32_SBRS= \
	"$(INTDIR)\axpy.sbr" \
	"$(INTDIR)\databox.sbr" \
	"$(INTDIR)\diff.sbr" \
	"$(INTDIR)\error.sbr" \
	"$(INTDIR)\flux.sbr" \
	"$(INTDIR)\getsubbox.sbr" \
	"$(INTDIR)\grid.sbr" \
	"$(INTDIR)\head.sbr" \
	"$(INTDIR)\pftappinit.sbr" \
	"$(INTDIR)\pftools.sbr" \
	"$(INTDIR)\printdatabox.sbr" \
	"$(INTDIR)\readdatabox.sbr" \
	"$(INTDIR)\region.sbr" \
	"$(INTDIR)\stats.sbr" \
	"$(INTDIR)\tools_io.sbr" \
	"$(INTDIR)\usergrid.sbr" \
	"$(INTDIR)\velocity.sbr"

"$(OUTDIR)\pftools.bsc" : "$(OUTDIR)" $(BSC32_SBRS)
    $(BSC32) @<<
  $(BSC32_FLAGS) $(BSC32_SBRS)
<<

LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib /nologo /subsystem:console /debug /machine:I386
# ADD LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib tcl76.lib tk42.lib /nologo /subsystem:console /debug /machine:I386 /out:"pftools.exe"
LINK32_FLAGS=kernel32.lib user32.lib gdi32.lib winspool.lib advapi32.lib\
 shell32.lib ole32.lib oleaut32.lib uuid.lib tcl76.lib tk42.lib /nologo\
 /subsystem:console /incremental:yes /pdb:"$(OUTDIR)/pftools.pdb" /debug\
 /machine:I386 /out:"pftools.exe" 
LINK32_OBJS= \
	"$(INTDIR)\axpy.obj" \
	"$(INTDIR)\databox.obj" \
	"$(INTDIR)\diff.obj" \
	"$(INTDIR)\error.obj" \
	"$(INTDIR)\flux.obj" \
	"$(INTDIR)\getsubbox.obj" \
	"$(INTDIR)\grid.obj" \
	"$(INTDIR)\head.obj" \
	"$(INTDIR)\pftappinit.obj" \
	"$(INTDIR)\pftools.obj" \
	"$(INTDIR)\printdatabox.obj" \
	"$(INTDIR)\readdatabox.obj" \
	"$(INTDIR)\region.obj" \
	"$(INTDIR)\stats.obj" \
	"$(INTDIR)\tools_io.obj" \
	"$(INTDIR)\usergrid.obj" \
	"$(INTDIR)\velocity.obj"

".\pftools.exe" : "$(OUTDIR)" $(DEF_FILE) $(LINK32_OBJS)
    $(LINK32) @<<
  $(LINK32_FLAGS) $(LINK32_OBJS)
<<

!ENDIF 

.c{$(CPP_OBJS)}.obj:
   $(CPP) $(CPP_PROJ) $<  

.cpp{$(CPP_OBJS)}.obj:
   $(CPP) $(CPP_PROJ) $<  

.cxx{$(CPP_OBJS)}.obj:
   $(CPP) $(CPP_PROJ) $<  

.c{$(CPP_SBRS)}.sbr:
   $(CPP) $(CPP_PROJ) $<  

.cpp{$(CPP_SBRS)}.sbr:
   $(CPP) $(CPP_PROJ) $<  

.cxx{$(CPP_SBRS)}.sbr:
   $(CPP) $(CPP_PROJ) $<  

################################################################################
# Begin Target

# Name "pftools - Win32 Release"
# Name "pftools - Win32 Debug"

!IF  "$(CFG)" == "pftools - Win32 Release"

!ELSEIF  "$(CFG)" == "pftools - Win32 Debug"

!ENDIF 

################################################################################
# Begin Source File

SOURCE=.\tools_io.c

!IF  "$(CFG)" == "pftools - Win32 Release"


"$(INTDIR)\tools_io.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "pftools - Win32 Debug"


"$(INTDIR)\tools_io.obj" : $(SOURCE) "$(INTDIR)"

"$(INTDIR)\tools_io.sbr" : $(SOURCE) "$(INTDIR)"


!ENDIF 

# End Source File
################################################################################
# Begin Source File

SOURCE=.\readdatabox.c
DEP_CPP_READD=\
	".\databox.h"\
	".\readdatabox.h"\
	".\tools_io.h"\
	"c:\program files\tcl\include\tcl.h"\
	

!IF  "$(CFG)" == "pftools - Win32 Release"


"$(INTDIR)\readdatabox.obj" : $(SOURCE) $(DEP_CPP_READD) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "pftools - Win32 Debug"


"$(INTDIR)\readdatabox.obj" : $(SOURCE) $(DEP_CPP_READD) "$(INTDIR)"

"$(INTDIR)\readdatabox.sbr" : $(SOURCE) $(DEP_CPP_READD) "$(INTDIR)"


!ENDIF 

# End Source File
################################################################################
# Begin Source File

SOURCE=.\printdatabox.c
DEP_CPP_PRINT=\
	".\databox.h"\
	".\printdatabox.h"\
	".\tools_io.h"\
	"c:\program files\tcl\include\tcl.h"\
	

!IF  "$(CFG)" == "pftools - Win32 Release"


"$(INTDIR)\printdatabox.obj" : $(SOURCE) $(DEP_CPP_PRINT) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "pftools - Win32 Debug"


"$(INTDIR)\printdatabox.obj" : $(SOURCE) $(DEP_CPP_PRINT) "$(INTDIR)"

"$(INTDIR)\printdatabox.sbr" : $(SOURCE) $(DEP_CPP_PRINT) "$(INTDIR)"


!ENDIF 

# End Source File
################################################################################
# Begin Source File

SOURCE=.\flux.c
DEP_CPP_FLUX_=\
	".\databox.h"\
	".\flux.h"\
	"c:\program files\tcl\include\tcl.h"\
	

!IF  "$(CFG)" == "pftools - Win32 Release"


"$(INTDIR)\flux.obj" : $(SOURCE) $(DEP_CPP_FLUX_) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "pftools - Win32 Debug"


"$(INTDIR)\flux.obj" : $(SOURCE) $(DEP_CPP_FLUX_) "$(INTDIR)"

"$(INTDIR)\flux.sbr" : $(SOURCE) $(DEP_CPP_FLUX_) "$(INTDIR)"


!ENDIF 

# End Source File
################################################################################
# Begin Source File

SOURCE=.\databox.c
DEP_CPP_DATAB=\
	".\databox.h"\
	"c:\program files\tcl\include\tcl.h"\
	

!IF  "$(CFG)" == "pftools - Win32 Release"


"$(INTDIR)\databox.obj" : $(SOURCE) $(DEP_CPP_DATAB) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "pftools - Win32 Debug"


"$(INTDIR)\databox.obj" : $(SOURCE) $(DEP_CPP_DATAB) "$(INTDIR)"

"$(INTDIR)\databox.sbr" : $(SOURCE) $(DEP_CPP_DATAB) "$(INTDIR)"


!ENDIF 

# End Source File
################################################################################
# Begin Source File

SOURCE=.\axpy.c
DEP_CPP_AXPY_=\
	".\databox.h"\
	".\diff.h"\
	".\error.h"\
	".\flux.h"\
	".\getsubbox.h"\
	".\head.h"\
	".\pftools.h"\
	".\printdatabox.h"\
	".\readdatabox.h"\
	".\stats.h"\
	".\velocity.h"\
	"c:\program files\tcl\include\tcl.h"\
	"c:\program files\tcl\include\tk.h"\
	"c:\program files\tcl\include\X11\X.h"\
	"c:\program files\tcl\include\X11\Xfuncproto.h"\
	"c:\program files\tcl\include\X11\Xlib.h"\
	

!IF  "$(CFG)" == "pftools - Win32 Release"


"$(INTDIR)\axpy.obj" : $(SOURCE) $(DEP_CPP_AXPY_) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "pftools - Win32 Debug"


"$(INTDIR)\axpy.obj" : $(SOURCE) $(DEP_CPP_AXPY_) "$(INTDIR)"

"$(INTDIR)\axpy.sbr" : $(SOURCE) $(DEP_CPP_AXPY_) "$(INTDIR)"


!ENDIF 

# End Source File
################################################################################
# Begin Source File

SOURCE=.\diff.c
DEP_CPP_DIFF_=\
	".\databox.h"\
	".\diff.h"\
	"c:\program files\tcl\include\tcl.h"\
	

!IF  "$(CFG)" == "pftools - Win32 Release"


"$(INTDIR)\diff.obj" : $(SOURCE) $(DEP_CPP_DIFF_) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "pftools - Win32 Debug"


"$(INTDIR)\diff.obj" : $(SOURCE) $(DEP_CPP_DIFF_) "$(INTDIR)"

"$(INTDIR)\diff.sbr" : $(SOURCE) $(DEP_CPP_DIFF_) "$(INTDIR)"


!ENDIF 

# End Source File
################################################################################
# Begin Source File

SOURCE=.\grid.c
DEP_CPP_GRID_=\
	".\databox.h"\
	".\file.h"\
	".\general.h"\
	".\grid.h"\
	".\load.h"\
	".\pfload_file.h"\
	".\readdatabox.h"\
	".\region.h"\
	".\usergrid.h"\
	"c:\program files\tcl\include\tcl.h"\
	

!IF  "$(CFG)" == "pftools - Win32 Release"


"$(INTDIR)\grid.obj" : $(SOURCE) $(DEP_CPP_GRID_) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "pftools - Win32 Debug"


"$(INTDIR)\grid.obj" : $(SOURCE) $(DEP_CPP_GRID_) "$(INTDIR)"

"$(INTDIR)\grid.sbr" : $(SOURCE) $(DEP_CPP_GRID_) "$(INTDIR)"


!ENDIF 

# End Source File
################################################################################
# Begin Source File

SOURCE=.\head.c
DEP_CPP_HEAD_=\
	".\databox.h"\
	".\head.h"\
	"c:\program files\tcl\include\tcl.h"\
	

!IF  "$(CFG)" == "pftools - Win32 Release"


"$(INTDIR)\head.obj" : $(SOURCE) $(DEP_CPP_HEAD_) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "pftools - Win32 Debug"


"$(INTDIR)\head.obj" : $(SOURCE) $(DEP_CPP_HEAD_) "$(INTDIR)"

"$(INTDIR)\head.sbr" : $(SOURCE) $(DEP_CPP_HEAD_) "$(INTDIR)"


!ENDIF 

# End Source File
################################################################################
# Begin Source File

SOURCE=.\region.c
DEP_CPP_REGIO=\
	".\databox.h"\
	".\file.h"\
	".\general.h"\
	".\grid.h"\
	".\load.h"\
	".\pfload_file.h"\
	".\readdatabox.h"\
	".\region.h"\
	".\usergrid.h"\
	"c:\program files\tcl\include\tcl.h"\
	

!IF  "$(CFG)" == "pftools - Win32 Release"


"$(INTDIR)\region.obj" : $(SOURCE) $(DEP_CPP_REGIO) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "pftools - Win32 Debug"


"$(INTDIR)\region.obj" : $(SOURCE) $(DEP_CPP_REGIO) "$(INTDIR)"

"$(INTDIR)\region.sbr" : $(SOURCE) $(DEP_CPP_REGIO) "$(INTDIR)"


!ENDIF 

# End Source File
################################################################################
# Begin Source File

SOURCE=.\stats.c
DEP_CPP_STATS=\
	".\databox.h"\
	".\stats.h"\
	"c:\program files\tcl\include\tcl.h"\
	

!IF  "$(CFG)" == "pftools - Win32 Release"


"$(INTDIR)\stats.obj" : $(SOURCE) $(DEP_CPP_STATS) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "pftools - Win32 Debug"


"$(INTDIR)\stats.obj" : $(SOURCE) $(DEP_CPP_STATS) "$(INTDIR)"

"$(INTDIR)\stats.sbr" : $(SOURCE) $(DEP_CPP_STATS) "$(INTDIR)"


!ENDIF 

# End Source File
################################################################################
# Begin Source File

SOURCE=.\velocity.c
DEP_CPP_VELOC=\
	".\databox.h"\
	".\velocity.h"\
	"c:\program files\tcl\include\tcl.h"\
	

!IF  "$(CFG)" == "pftools - Win32 Release"


"$(INTDIR)\velocity.obj" : $(SOURCE) $(DEP_CPP_VELOC) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "pftools - Win32 Debug"


"$(INTDIR)\velocity.obj" : $(SOURCE) $(DEP_CPP_VELOC) "$(INTDIR)"

"$(INTDIR)\velocity.sbr" : $(SOURCE) $(DEP_CPP_VELOC) "$(INTDIR)"


!ENDIF 

# End Source File
################################################################################
# Begin Source File

SOURCE=.\pftools.c
DEP_CPP_PFTOO=\
	".\databox.h"\
	".\diff.h"\
	".\error.h"\
	".\flux.h"\
	".\getsubbox.h"\
	".\head.h"\
	".\pftools.h"\
	".\printdatabox.h"\
	".\readdatabox.h"\
	".\stats.h"\
	".\velocity.h"\
	"c:\program files\tcl\include\tcl.h"\
	"c:\program files\tcl\include\tk.h"\
	"c:\program files\tcl\include\X11\X.h"\
	"c:\program files\tcl\include\X11\Xfuncproto.h"\
	"c:\program files\tcl\include\X11\Xlib.h"\
	

!IF  "$(CFG)" == "pftools - Win32 Release"


"$(INTDIR)\pftools.obj" : $(SOURCE) $(DEP_CPP_PFTOO) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "pftools - Win32 Debug"


"$(INTDIR)\pftools.obj" : $(SOURCE) $(DEP_CPP_PFTOO) "$(INTDIR)"

"$(INTDIR)\pftools.sbr" : $(SOURCE) $(DEP_CPP_PFTOO) "$(INTDIR)"


!ENDIF 

# End Source File
################################################################################
# Begin Source File

SOURCE=.\usergrid.c
DEP_CPP_USERG=\
	".\databox.h"\
	".\file.h"\
	".\general.h"\
	".\grid.h"\
	".\load.h"\
	".\pfload_file.h"\
	".\readdatabox.h"\
	".\region.h"\
	".\usergrid.h"\
	"c:\program files\tcl\include\tcl.h"\
	

!IF  "$(CFG)" == "pftools - Win32 Release"


"$(INTDIR)\usergrid.obj" : $(SOURCE) $(DEP_CPP_USERG) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "pftools - Win32 Debug"


"$(INTDIR)\usergrid.obj" : $(SOURCE) $(DEP_CPP_USERG) "$(INTDIR)"

"$(INTDIR)\usergrid.sbr" : $(SOURCE) $(DEP_CPP_USERG) "$(INTDIR)"


!ENDIF 

# End Source File
################################################################################
# Begin Source File

SOURCE=.\error.c
DEP_CPP_ERROR=\
	".\databox.h"\
	".\error.h"\
	"c:\program files\tcl\include\tcl.h"\
	

!IF  "$(CFG)" == "pftools - Win32 Release"


"$(INTDIR)\error.obj" : $(SOURCE) $(DEP_CPP_ERROR) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "pftools - Win32 Debug"


"$(INTDIR)\error.obj" : $(SOURCE) $(DEP_CPP_ERROR) "$(INTDIR)"

"$(INTDIR)\error.sbr" : $(SOURCE) $(DEP_CPP_ERROR) "$(INTDIR)"


!ENDIF 

# End Source File
################################################################################
# Begin Source File

SOURCE=.\pftappinit.c
DEP_CPP_PFTAP=\
	".\databox.h"\
	".\diff.h"\
	".\error.h"\
	".\flux.h"\
	".\getsubbox.h"\
	".\head.h"\
	".\pftools.h"\
	".\printdatabox.h"\
	".\readdatabox.h"\
	".\stats.h"\
	".\tools_io.h"\
	".\velocity.h"\
	"c:\program files\tcl\include\tcl.h"\
	"c:\program files\tcl\include\tk.h"\
	"c:\program files\tcl\include\X11\X.h"\
	"c:\program files\tcl\include\X11\Xfuncproto.h"\
	"c:\program files\tcl\include\X11\Xlib.h"\
	

!IF  "$(CFG)" == "pftools - Win32 Release"


"$(INTDIR)\pftappinit.obj" : $(SOURCE) $(DEP_CPP_PFTAP) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "pftools - Win32 Debug"


"$(INTDIR)\pftappinit.obj" : $(SOURCE) $(DEP_CPP_PFTAP) "$(INTDIR)"

"$(INTDIR)\pftappinit.sbr" : $(SOURCE) $(DEP_CPP_PFTAP) "$(INTDIR)"


!ENDIF 

# End Source File
################################################################################
# Begin Source File

SOURCE=.\getsubbox.c
DEP_CPP_GETSU=\
	".\databox.h"\
	".\getsubbox.h"\
	"c:\program files\tcl\include\tcl.h"\
	

!IF  "$(CFG)" == "pftools - Win32 Release"


"$(INTDIR)\getsubbox.obj" : $(SOURCE) $(DEP_CPP_GETSU) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "pftools - Win32 Debug"


"$(INTDIR)\getsubbox.obj" : $(SOURCE) $(DEP_CPP_GETSU) "$(INTDIR)"

"$(INTDIR)\getsubbox.sbr" : $(SOURCE) $(DEP_CPP_GETSU) "$(INTDIR)"


!ENDIF 

# End Source File
# End Target
# End Project
################################################################################
