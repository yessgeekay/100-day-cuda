#|/bin/bash

CWD=$(pwd)

for (( i=0; i<4; i++ )); do
  echo "Executing output$i"
  export CUDA_VISIBLE_DEVICES=3
  COMMAND="$CWD/output$i.o"
  echo $COMMAND >> $CWD/output.txt
  $COMMAND >> $CWD/output.txt
  echo "-------------------------------------------------" >> $CWD/output.txt
done
