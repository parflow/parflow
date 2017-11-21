#!/bin/bash
when-changed -r -s $PARFLOW_DIR/ -c bash -c "killall do.sh ; ./do.sh & exit 0"



