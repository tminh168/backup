#!/bin/bash

restart=""
while true;
do
    timeout 605 python3 /home/mendel/coral/DFM_counter/main_counter.py $restart
    restart="restart"
    read -t 0.3 -n 1 k
    [[ "$k" == 's' ]] && break
done
