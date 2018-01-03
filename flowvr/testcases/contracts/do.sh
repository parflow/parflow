#!/bin/bash
rm $PROBLEMNAME*.out.*
export PROBLEMNAME=default_richards
$PARFLOW_DIR/bin/parflowvr/do.sh $PROBLEMNAME --batch-mode >log
echo .
echo .
ERROR=0
if [ "$(grep -a 'YES' log | wc -l)" != "6" ];
then
  echo Error! Did not found 6 times YES in log!
  ERROR=1
fi

if [ "$(grep -ia Error log | wc -l)" != "0" ];
then
  echo Error! Did find Errors in log!
  ERROR=1
fi

if [ "$ERROR" == "0" ];
then
  echo passed tests!
fi

exit $ERROR
