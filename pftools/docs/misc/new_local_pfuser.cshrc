
setenv PARFLOW_DIR          /home/casc/parflow/exe.`uname`
setenv PARFLOW_HELP         /home/casc/parflow/docs
setenv PARFLOW_HTML_VIEWER  /usr/local/bin/netscape3
setenv XUSERFILESEARCHPATH  $PARFLOW_DIR/bin/%N

setenv AVS_PATH             /usr/local/avs
setenv AVS_HELP_PATH        $PARFLOW_DIR/avs/help

#Next line should no longer be needed.  Keep it in till sure.
#setenv OPENWINHOME          /usr/openwin

if ( `uname` == "IRIX64" ) then
   set path=(. $PARFLOW_DIR/bin $path)
   stty dec
else
   set path=(.  $PARFLOW_DIR/bin $path)
endif

