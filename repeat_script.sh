#!/bin/bash

COMMAND=$1
REPEAT=10

for ((i=1; i<=REPEAT; i++))
do
    echo "Repeat $i"
    bash $COMMAND
done
