# Contributing to ParFlow

__[under construction]__

- Before committing make sure to run `bin/pfformat` from the projects root.
Alternatively add this to your `parflow/.git/hooks/pre-commit` file:
```bash
#!/bin/bash
p=`git rev-parse --show-toplevel`
cd $p

files=$(git status --porcelain | grep '^M .*\.[ch]$' | cut  -d ' ' -f 3)
for i in $files
do
  uncrustify --check -l C -c bin/parflow.cfg $i
  RESULT=$?
  if [ "$RESULT" != "0" ]; then
    echo $RESULT
    exit $RESULT
  fi
done
```
