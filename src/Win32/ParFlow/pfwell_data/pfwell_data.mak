# Microsoft Developer Studio Generated NMAKE File, Based on pfwell_data.dsp
!IF "$(CFG)" == ""
CFG=pfwell_data - Win32 Debug
!MESSAGE No configuration specified. Defaulting to pfwell_data - Win32 Debug.
!ENDIF 

!IF "$(CFG)" != "pfwell_data - Win32 Release" && "$(CFG)" != "pfwell_data - Win32 Debug"
!MESSAGE Invalid configuration "$(CFG)" specified.
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "pfwell_data.mak" CFG="pfwell_data - Win32 Debug"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "pfwell_data - Win32 Release" (based on "Win32 (x86) Console Application")
!MESSAGE "pfwell_data - Win32 Debug" (based on "Win32 (x86) Console Application")
!MESSAGE 
!ERROR An invalid configuration is specified.
!ENDIF 

!IF "$(OS)" == "Windows_NT"
NULL=
!ELSE 
NULL=nul
!ENDIF 

!IF  "$(CFG)" == "pfwell_data - Win32 Release"

OUTDIR=.\Release
INTDIR=.\Release
# Begin Custom Macros
OutDir=.\Release
# End Custom Macros

ALL : "$(OUTDIR)\pfwell_data.exe"


CLEAN :
	-@erase "$(INTDIR)\pfwell_data.obj"
	-@erase "$(INTDIR)\vc60.idb"
	-@erase "$(INTDIR)\well.obj"
	-@erase "$(OUTDIR)\pfwell_data.exe"

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

F90=df.exe
CPP=cl.exe
CPP_PROJ=/nologo /ML /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_CONSOLE" /D "_MBCS" /Fp"$(INTDIR)\pfwell_data.pch" /YX /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

.c{$(INTDIR)}.obj::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cpp{$(INTDIR)}.obj::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cxx{$(INTDIR)}.obj::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.c{$(INTDIR)}.sbr::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cpp{$(INTDIR)}.sbr::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cxx{$(INTDIR)}.sbr::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

RSC=rc.exe
BSC32=bscmake.exe
BSC32_FLAGS=/nologo /o"$(OUTDIR)\pfwell_data.bsc" 
BSC32_SBRS= \
	
LINK32=link.exe
LINK32_FLAGS=kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /incremental:no /pdb:"$(OUTDIR)\pfwell_data.pdb" /machine:I386 /out:"$(OUTDIR)\pfwell_data.exe" 
LINK32_OBJS= \
	"$(INTDIR)\pfwell_data.obj" \
	"$(INTDIR)\well.obj"

"$(OUTDIR)\pfwell_data.exe" : "$(OUTDIR)" $(DEF_FILE) $(LINK32_OBJS)
    $(LINK32) @<<
  $(LINK32_FLAGS) $(LINK32_OBJS)
<<

!ELSEIF  "$(CFG)" == "pfwell_data - Win32 Debug"

OUTDIR=.\Debug
INTDIR=.\Debug
# Begin Custom Macros
OutDir=.\Debug
# End Custom Macros

ALL : "$(OUTDIR)\pfwell_data.exe" "$(OUTDIR)\pfwell_data.bsc"


CLEAN :
	-@erase "$(INTDIR)\pfwell_data.obj"
	-@erase "$(INTDIR)\pfwell_data.sbr"
	-@erase "$(INTDIR)\vc60.idb"
	-@erase "$(INTDIR)\vc60.pdb"
	-@erase "$(INTDIR)\well.obj"
	-@erase "$(INTDIR)\well.sbr"
	-@erase "$(OUTDIR)\pfwell_data.bsc"
	-@erase "$(OUTDIR)\pfwell_data.exe"
	-@erase "$(OUTDIR)\pfwell_data.ilk"
	-@erase "$(OUTDIR)\pfwell_data.pdb"

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

F90=df.exe
CPP=cl.exe
CPP_PROJ=/nologo /MLd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_CONSOLE" /D "_MBCS" /FR"$(INTDIR)\\" /Fp"$(INTDIR)\pfwell_data.pch" /YX /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

.c{$(INTDIR)}.obj::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cpp{$(INTDIR)}.obj::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cxx{$(INTDIR)}.obj::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.c{$(INTDIR)}.sbr::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cpp{$(INTDIR)}.sbr::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cxx{$(INTDIR)}.sbr::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

RSC=rc.exe
BSC32=bscmake.exe
BSC32_FLAGS=/nologo /o"$(OUTDIR)\pfwell_data.bsc" 
BSC32_SBRS= \
	"$(INTDIR)\pfwell_data.sbr" \
	"$(INTDIR)\well.sbr"

"$(OUTDIR)\pfwell_data.bsc" : "$(OUTDIR)" $(BSC32_SBRS)
    $(BSC32) @<<
  $(BSC32_FLAGS) $(BSC32_SBRS)
<<

LINK32=link.exe
LINK32_FLAGS=kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:console /incremental:yes /pdb:"$(OUTDIR)\pfwell_data.pdb" /debug /machine:I386 /out:"$(OUTDIR)\pfwell_data.exe" /pdbtype:sept 
LINK32_OBJS= \
	"$(INTDIR)\pfwell_data.obj" \
	"$(INTDIR)\well.obj"

"$(OUTDIR)\pfwell_data.exe" : "$(OUTDIR)" $(DEF_FILE) $(LINK32_OBJS)
    $(LINK32) @<<
  $(LINK32_FLAGS) $(LINK32_OBJS)
<<

!ENDIF 


!IF "$(NO_EXTERNAL_DEPS)" != "1"
!IF EXISTS("pfwell_data.dep")
!INCLUDE "pfwell_data.dep"
!ELSE 
!MESSAGE Warning: cannot find "pfwell_data.dep"
!ENDIF 
!ENDIF 


!IF "$(CFG)" == "pfwell_data - Win32 Release" || "$(CFG)" == "pfwell_data - Win32 Debug"
SOURCE=..\..\..\tools\pfwell_data.c

!IF  "$(CFG)" == "pfwell_data - Win32 Release"


"$(INTDIR)\pfwell_data.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "pfwell_data - Win32 Debug"


"$(INTDIR)\pfwell_data.obj"	"$(INTDIR)\pfwell_data.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

SOURCE=..\..\..\tools\well.c

!IF  "$(CFG)" == "pfwell_data - Win32 Release"


"$(INTDIR)\well.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "pfwell_data - Win32 Debug"


"$(INTDIR)\well.obj"	"$(INTDIR)\well.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 


!ENDIF 

