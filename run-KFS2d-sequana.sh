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

module load sequana/current
module load intel_psxe/2019_sequana
module load python/3.8.2_sequana 
module load openmpi/gnu/2.1.6-gcc-8.3-cuda_sequana
source /scratch/cenapadrjsd/rpsouto/sequana/usr/local/spack/git/spack/share/spack/setup-env.sh
export SPACK_USER_CONFIG_PATH=/scratch/cenapadrjsd/rpsouto/.spack/v0.17.1
spack load openblas
spack load gperftools
export MALLOCSTATS=1

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

executavel=KFS2d
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_SCHEDULE=DYNAMIC

#Executa o KFS2d 
#srun  -N 1 -c $SLURM_CPUS_PER_TASK $EXEC $assimType $gridX $gridY $timeStep $freqObsT $freqObsX $freqObsY $percNoise $neuronNumber 2>&1 | tee output.log 
srun -N 1 -n 1 -c $SLURM_CPUS_PER_TASK $executavel $assimType $gridX $gridY $timeStep $freqObsT $freqObsX $freqObsY $percNoise $neuronNumber

if [[ ${assimType} -eq 1 ]]; then
  echo "Copiando o resultado da assimilacao por FK e os dados para treinamento da RNA."
  mkdir -p ${resultsdir}/${outputdir}/full/omp-${SLURM_CPUS_PER_TASK}/job-${SLURM_JOB_ID}
  cp slurm-${SLURM_JOB_ID}.out ${resultsdir}/${outputdir}/full/omp-${SLURM_CPUS_PER_TASK}/job-${SLURM_JOB_ID}/
  cp output.log ${resultsdir}/${outputdir}/full/omp-${SLURM_CPUS_PER_TASK}/job-${SLURM_JOB_ID}/
  cp output/full/*.out ${resultsdir}/${outputdir}/full/omp-${SLURM_CPUS_PER_TASK}/job-${SLURM_JOB_ID}/
  mkdir -p ${resultsdir}/${outputdir}/training/omp-${SLURM_CPUS_PER_TASK}/job-${SLURM_JOB_ID}
  cp output/training/*.out ${resultsdir}/${outputdir}/training/omp-${SLURM_CPUS_PER_TASK}/job-${SLURM_JOB_ID}/
fi

#if [[ ${assimType} -eq 2 && -d ${resultsdir}/${outputdir} ]]; then
if [[ ${assimType} -eq 2 ]]; then
  if [[ ! -d $resultsdir/$outputdir ]]; then
     mkdir -p $resultsdir/$outputdir
  fi
  echo "Copiando o resultado da assimilacao de FK emulada por RNA."
  mkdir -p ${resultsdir}/${outputdir}/full/omp-${SLURM_CPUS_PER_TASK}/job-${SLURM_JOB_ID}
  cp output/full/qObservExpA.out       ${resultsdir}/${outputdir}/full/omp-${SLURM_CPUS_PER_TASK}/job-${SLURM_JOB_ID}/
  cp output/full/qModelExpA.out        ${resultsdir}/${outputdir}/full/omp-${SLURM_CPUS_PER_TASK}/job-${SLURM_JOB_ID}/
  cp output/full/qAnalysisExpA_RNA.out ${resultsdir}/${outputdir}/full/omp-${SLURM_CPUS_PER_TASK}/job-${SLURM_JOB_ID}/
  cp slurm-${SLURM_JOB_ID}.out ${resultsdir}/${outputdir}/full/omp-${SLURM_CPUS_PER_TASK}/job-${SLURM_JOB_ID}/
  cp output.log ${resultsdir}/${outputdir}/full/omp-${SLURM_CPUS_PER_TASK}/job-${SLURM_JOB_ID}/
  mkdir -p ${resultsdir}/${outputdir}/training/omp-${SLURM_CPUS_PER_TASK}/job-${SLURM_JOB_ID}
  cp output/training/*.out ${resultsdir}/${outputdir}/training/omp-${SLURM_CPUS_PER_TASK}/job-${SLURM_JOB_ID}/
fi


