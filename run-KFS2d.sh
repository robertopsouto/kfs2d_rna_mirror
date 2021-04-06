#!/bin/bash
#Uso: 
# ./run-KFS2d.sh <assimType> <gridX> <gridY> <timeStep> <freqObsT> <freqObsX> <freqObsY> <neuronNumber>
#
#Uso com Filtro de Kalman: 
# .run-KFS2d.sh 1 10 10 60 10 2 2 10
#
#Uso com RNA: 
# .run-KFS2d.sh 2 10 10 60 10 2 2 10

assimType=${1}
gridX=${2}
gridY=${3} 
timeStep=${4} 
freqObsT=${5} 
freqObsX=${6} 
freqObsY=${7} 
neuronNumber=${8}

#DATE=$(date +%Y-%m-%d-%H%M%S)
#starttime=starttime_${DATE}

if [[ ! -d resultados ]]; then
  mkdir resultados
fi

resultdir="output-gridX_$gridX-gridY_$gridY-timestep_$timeStep-freqObsT_$freqObsT-freqObsX_$freqObsX-freqObsY_$freqObsY-neuronNumber_$neuronNumber"
if [[ ${assimType} -eq 1 && -d resultados/$resultdir ]]; then
  echo "Resultado com FK já gerado com estes parametros!"
  exit 1
fi

#Executa o KFS2d 
./KFS2d $assimType $gridX $gridY $timeStep $freqObsT $freqObsX $freqObsY $neuronNumber 2>&1 | tee output.log 

if [[ ${assimType} -eq 1 ]]; then
  echo "Copiando o resultado da assimilacao por FK e os dados para treinamento da RNA."
  cp -r -p output/ resultados/${resultdir}
  mv output.log resultados/${resultdir}/output_FK.log
fi

if [[ ${assimType} -eq 2 && -d resultados/$resultdir ]]; then
  echo "Copiando o resultado da assimilacao de FK emulada por RNA."
  cp output/full/qAnalysisExpA_RNA.out resultados/$resultdir/full/
  cp output/computingANNTime.out resultados/$resultdir  
  mv output.log resultados/${resultdir}/output_RNA.log
fi


