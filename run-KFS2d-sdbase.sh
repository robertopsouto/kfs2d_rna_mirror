#!/bin/bash
#SBATCH --nodes=1                      #Numero de Nós
#SBATCH --ntasks=1                     #Numero total de tarefas MPI
#SBATCH --cpus-per-task=16
#SBATCH -p nvidia_small                        #Fila (partition) a ser utilizada
#SBATCH -J KFS2d                        #Nome job
#Uso: 
# ./run-KFS2d.sh <assimType> <gridX> <gridY> <timeStep> <freqObsT> <freqObsX> <freqObsY> <percNoise> <neuronNumber>
#
#Uso com Filtro de Kalman: 
# ./run-KFS2d.sh 1 10 10 60 10 2 2 0.2 10
#
#Uso com RNA: 
# ./run-KFS2d.sh 2 10 10 60 10 2 2 0.2 10
#sbatch -p cpu_dev -c 16 ./run-KFS2d-slurm.sh 2 1280 1280 200 10 256 256 0.1 10
#sbatch -p cpu_dev -c 4  ./run-KFS2d-slurm.sh 2 40 40 500 10 8 8 0.1 10

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

#module load python/3.8.2
#module load gcc/8.3
#source /scratch/cenapadrjsd/rpsouto/usr/local/spack/git/spack/share/spack/setup-env.sh
#spack load -r openblas

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
rm output/*.out
rm output/full/*.out
rm output/training/*.out

executavel=/scratch/cenapadrjsd/rpsouto/projetos/g-assimila/gitlab/kfs2d_rna/KFS2d
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

ldd $executavel

#Executa o KFS2d 
#srun  -N 1 -c $SLURM_CPUS_PER_TASK $EXEC $assimType $gridX $gridY $timeStep $freqObsT $freqObsX $freqObsY $percNoise $neuronNumber 2>&1 | tee output.log 
srun -N 1 -n 1 -c $SLURM_CPUS_PER_TASK $executavel $assimType $gridX $gridY $timeStep $freqObsT $freqObsX $freqObsY $percNoise $neuronNumber

if [[ ${assimType} -eq 1 ]]; then
  echo "Copiando o resultado da assimilacao por FK e os dados para treinamento da RNA."
  cp -r -p output/ ${resultsdir}/$outputdir
  cp output.log ${resultsdir}/${outputdir}/output_FK.log
fi

if [[ ${assimType} -eq 2 && -d ${resultsdir}/$outputdir ]]; then
  echo "Copiando o resultado da assimilacao de FK emulada por RNA."
  #cp output/full/qAnalysisExpA_RNA.out ${resultsdir}/${outputdir}/full/qAnalysisExpA_RNA_omp-${SLURM_CPUS_PER_TASK}_job-${SLURM_JOB_ID}.out
  if [[ -L ${resultsdir}/${outputdir}/full/qAnalysisExpA_RNA.out ]]; then
	  rm ${resultsdir}/${outputdir}/full/qAnalysisExpA_RNA.out
  fi
  #ln -s -r ${resultsdir}/${outputdir}/full/qAnalysisExpA_RNA_omp-${SLURM_CPUS_PER_TASK}_job-${SLURM_JOB_ID}.out ${resultsdir}/${outputdir}/full/qAnalysisExpA_RNA.out
  cp output.log ${resultsdir}/${outputdir}/output_RNA-neuronNumber_omp-${SLURM_CPUS_PER_TASK}_job-${SLURM_JOB_ID}.log
fi

