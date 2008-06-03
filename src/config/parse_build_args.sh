#! /usr/local/bin/bash
#BHEADER***********************************************************************
# (c) 1995   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision: 1.1.1.1 $
#EHEADER***********************************************************************

SHELL=/bin/sh
export SHELL

PF_CONFIG=""
#=============================================================================
#
# Loop to parse users arguments
#
#=============================================================================
while [ "$*" != "" ]
do
case $1 in
  -h|-help) 
	echo "$0 [-O] [-g] [install|clean|all] "
	echo 
	echo "Where:"
	echo "-O              optimize"
	echo "-g              debug"
	echo "-p              profiling"
	echo "-time           add timing"
	echo " all	      compiles but does not install"
	echo " install	      builds and installs"
	echo " clean	      cleans temporary build files (ie .o)"
	echo " veryclean      cleans everything"
	echo "arch            is the architecture of the machine"
	echo "comm            is the message passing package to use"
       exit 0;;
  -config)
	PF_CONFIG=$2
	shift
	shift;;
  -O) 
       PF_OPTIMIZE_FLAG=1
       shift;;
  -g) 
       PF_DEBUG_FLAG=1
       shift;;
  -p) 
       PF_PROFILE_FLAG=1
       shift;;
  -time) 
       PF_TIMING_FLAG=1
       shift;;
  all) shift;;
  install) 
	PF_INSTALL_FLAG=1
	shift;;
  clean)
	PF_CLEAN_FLAG=1
	shift;;
  veryclean)
	PF_CLEAN_FLAG=1
	PF_VERYCLEAN_FLAG=1
	shift;;
  docs)
	PF_DOCS_FLAG=1
	shift;;
  -arch)
       PF_ARCH=$2
       shift 
       shift;;
  -comm)
       PF_COMM=$2
       shift
       shift;;
  *)
       echo "Unknown option: $1"
       exit;;
esac
done

PF_ALL_FLAG=""
if [ -z "$PF_CLEAN_FLAG" ]
then
	PF_ALL_FLAG=1
fi

export PF_OPTIMIZE_FLAG PF_DEBUG_FLAG PF_INSTALL_FLAG PF_CLEAN_FLAG PF_VERYCLEAN_FLAG PF_ALL_FLAG PF_DOCS_FLAG PF_CONFIG
export PF_CC_USER_OPTS

#
# SGS add check for the config file here
#

#=============================================================================
#
# Record the last arch build on
#
#=============================================================================
if [ -f .current_arch ] 
then
	PF_OLD_ARCH=`cat .current_arch`
else
	PF_OLD_ARCH="NONE"
fi
echo "$PF_ARCH" > .current_arch

if [ -z "$PF_ARCH" ]
then
	echo "I don't know what architecture I am on, please"
	echo "specify a config file with the -config option"
fi

. $PARFLOW_SRC/config/defaults.config

# If user specifies config file use it else look for one
if [ -z "$PF_CONFIG" ] 
then
	if [ -f $PARFLOW_SRC/config/$PF_ARCH.$PF_COMM.config ]
	then
		. $PARFLOW_SRC/config/$PF_ARCH.$PF_COMM.config
        elif [ -f  $PARFLOW_SRC/config/$PF_ARCH.config ]
	then
		. $PARFLOW_SRC/config/$PF_ARCH.config
	else
		echo "Can't find config file: using the defaults"
	fi
else
    if [ -f $PF_CONFIG ]
    then
	. $PF_CONFIG
    else
	echo "Can't find the config file $PF_CONFIG"
	exit 1
    fi
fi

export PF_OLD_ARCH PF_ARCH


