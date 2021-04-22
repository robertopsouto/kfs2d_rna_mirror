#!/bin/bash
#Uso: 
# ./run-KFS2d.sh <assimType> <gridX> <gridY> <timeStep> <freqObsT> <freqObsX> <freqObsY> <percNoise> <neuronNumber>
#
#Uso com Filtro de Kalman: 
# .run-KFS2d.sh 1 10 10 60 10 2 2 0.2 10
#
#Uso com RNA: 
# .run-KFS2d.sh 2 10 10 60 10 2 2 0.2 10

assimType=${1}
gridX=${2}
gridY=${3} 
timeStep=${4} 
freqObsT=${5} 
freqObsX=${6} 
freqObsY=${7} 
percNoise=${8} 
neuronNumber=${9}

#DATE=$(date +%Y-%m-%d-%H%M%S)
#starttime=starttime_${DATE}

resultsdir=resultados/percNoise_$percNoise

if [[ ! -d $resultsdir ]]; then
  mkdir -p $resultsdir
fi

outputdir="output-gridX_$gridX-gridY_$gridY-timestep_$timeStep-freqObsT_$freqObsT-freqObsX_$freqObsX-freqObsY_$freqObsY"
if [[ ${assimType} -eq 1 && -d ${resultsdir}/$outputdir ]]; then
  echo "Resultado com FK já gerado com estes parametros!"
  exit 1
fi

#Apaga os resultados anteriores
#rm output/*.out
#rm output/full/*.out
#rm output/training/*.out

#Executa o KFS2d 
./KFS2d $assimType $gridX $gridY $timeStep $freqObsT $freqObsX $freqObsY $percNoise $neuronNumber 2>&1 | tee output.log 

if [[ ${assimType} -eq 1 ]]; then
  echo "Copiando o resultado da assimilacao por FK e os dados para treinamento da RNA."
  cp -r -p output/ ${resultsdir}/$outputdir
  cp output.log ${resultsdir}/${outputdir}/output_FK.log
fi

if [[ ${assimType} -eq 2 && -d ${resultsdir}/$outputdir ]]; then
  echo "Copiando o resultado da assimilacao de FK emulada por RNA."
  cp output/full/qAnalysisExpA_RNA.out ${resultsdir}/${outputdir}/full/qAnalysisExpA_RNA-neuronNumber_${neuronNumber}.out
  if [[ -L ${resultsdir}/${outputdir}/full/qAnalysisExpA_RNA.out ]]; then
	  rm ${resultsdir}/${outputdir}/full/qAnalysisExpA_RNA.out
  fi
  ln -s -r ${resultsdir}/${outputdir}/full/qAnalysisExpA_RNA-neuronNumber_${neuronNumber}.out ${resultsdir}/${outputdir}/full/qAnalysisExpA_RNA.out
  cp output/computingANNTime.out ${resultsdir}/${outputdir}/computingANNTime-neuronNumber_${neuronNumber}.out
  cp output.log ${resultsdir}/${outputdir}/output_RNA-neuronNumber_${neuronNumber}.log
fi


