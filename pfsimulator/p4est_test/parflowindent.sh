#! /bin/bash

# Recall that code enclosed by  /* *INDENT-OFF* */  and
#  /* *INDENT-ON */ wont be formated.

#We will use a modified version of the original Berkley style
#whose settings are (see manpage):
# "$INDENT \
#	-nbad -nbap -bbo -bc -br -brs -c33 -cd33 -cdb -ce -ci4 -cli0
#       -cp33 -di16 -fc1 -fca -hnl -i4 -ip4 -l75 -lp -npcs -nprs -psl
#	-saf -sai -saw -sc -nsob -nss -ts8"

# We will introduce the changes:
# No blank line after declaration -nbad
# Blanck line after procedure -bap
# No tabs -nut
# Swallow optional blanck lines

INDENT_OPTIONS="-nbad -bap -bbo -nbc -br -brs -c33 -cd33 -cdb -ce -ci4 -cli0
            	-cp33 -di16 -fc1 -fca -hnl -i4 -ip4 -l75 -lp -npcs -nprs -psl
	        -saf -sai -saw -sc -nsob -nss -ts8 -nut -sob"

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
