#!/bin/bash -x

errors=0
p=$PWD
for t in contracts netcdfwriter steering-C steering-Python coordinates MergeGridMessages ;
do
  cd $p
  cd $t
  ./do.sh
  errors=$errors+$?
  echo .
  sleep 2
done
echo .
echo .
echo .
echo .
echo .
errorcount=`echo $errors | bc`
echo -------- $errorcount errors! -------

exit $errorcount
