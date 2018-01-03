# Contributing to ParFlow

__[under construction]__

- Before committing make sure to run `bin/pfformat` from the project's root.
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


## Coding Style
- 2 space indentation
- CamelCase for external *function names* and *struct typedef's* starting uppercase:
```C
DoSomething();
SuperCoolStruct object;
```

see also this issue:
https://github.com/parflow/parflow/issues/63

## Documentation
- there is the parflow-manual.pdf
- there is also the outdated developer manual that is useful.


