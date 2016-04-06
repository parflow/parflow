#! /bin/bash

# This is the default style for indent 2.2.10 (see man page)
#
#"$INDENT" \
#    -nbad -bap -nbc -bbo -bl -bli2 -bls -ncdb -nce -cp1 -cs -di2
#    -ndj -nfc1 -nfca -hnl -i2 -ip5 -lp -pcs -nprs -psl -saf -sai
#    -saw -nsc -nsob
#    "$@"

# Recall that code enclosed by  /* *INDENT-OFF* */  and
#  /* *INDENT-ON */ wont be formated. 

# blank lines after declarations ( -bad )
# blank lines after procedures   ( -bap )
# comment delimiters on blank lines ( -cdb -sc )
# braces indent 0 ( -bli0 ) 
# braces on if line, else after ( -br -nce )
# declarations set to indent 20 ( -di20 )
# use spaces instead of tabs ( -nut )
# swallow optional blank lines ( -sob )
# Put the type of a procedure on the line before its name ( -psl )

INDENT_OPTIONS="-bad -bap -cdb -sc -bli0 -br -nce -di20 -nut -sob -psl"

INDENT=`which gnuindent 2> /dev/null`
if test -z "$INDENT" ; then
	INDENT=`which gindent`
fi
if test -z "$INDENT" ; then
	INDENT=`which indent`
fi

for arg in "$@" ; do
  if [ "x$arg" == "x-o" ]; then
    WANTSOUT=1
  fi
done
if [ -z "$WANTSOUT" ]; then
  for NAME in "$@" ; do
    $INDENT $INDENT_OPTIONS "$NAME"
  done
else
  $INDENT $INDENT_OPTIONS $@
fi
