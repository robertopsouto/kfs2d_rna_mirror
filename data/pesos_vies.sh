#!/bin/bash
# Uso:
# ./pesos_vies.sh ann##.best (mpca file)

file=${1}

n=`sed -n 4p ${file}`

sed -n -e 6p -e 7p ${file} > wqcoExpA.dat

sed -n 9p ${file} > bqcoExpA.dat

x=$((10 + n)) 
y=$((12 + n))

sed -n 11,${x}p ${file} > wqcsExpA.dat

sed -n ${y}p ${file} > bqcsExpA.dat

