cd %PARFLOW_SRC%
mkdir %PARFLOW_DIR%
mkdir %PARFLOW_DIR%\bin
mkdir %PARFLOW_DIR%\test

cd %PARFLOW_SRC%\Win32\ParFlow
rem Comment this line to build debug version
set CFG=ParFlow - Win32 Release
nmake -f ParFlow.mak
set CFG=

rem Uncomment the line for debug/release
rem copy Debug\*.exe %PARFLOW_DIR%\bin
copy Release\*.exe %PARFLOW_DIR%\bin

cd %PARFLOW_SRC%\Win32\ParFlow\pfwell_cat
nmake -f pfwell_cat.mak
copy Debug\*.exe %PARFLOW_DIR%\bin

copy Debug\*.exe %PARFLOW_DIR%\bin
cd %PARFLOW_SRC%\Win32\ParFlow\pfwell_data
nmake -f pfwell_data.mak
copy Debug\*.exe %PARFLOW_DIR%\bin

cd %PARFLOW_SRC%\tools
nmake -f Makefile.tcllib.mak

copy parflow.dll %PARFLOW_DIR%\bin
copy *.tcl %PARFLOW_DIR%\bin
cd %PARFLOW_SRC%\test
copy * %PARFLOW_DIR%\test
cd %PARFLOW_SRC%\amps\win32
copy *.tcl %PARFLOW_DIR%\bin
copy *.bat %PARFLOW_DIR%\bin
cd %PARFLOW_SRC%\tools
copy *.tcl %PARFLOW_DIR%\bin
copy *.opt %PARFLOW_DIR%\bin
copy *.bat %PARFLOW_DIR%\bin
copy *.xbm %PARFLOW_DIR%\bin
copy *.sh  %PARFLOW_DIR%\bin
cd %PARFLOW_SRC%

rem cd %PARFLOW_DIR%\test
