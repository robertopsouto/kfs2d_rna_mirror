PROGRAM KFShallow2D
!****************************************************************************************
! Os componentes basicos para sistemas operacionais de previsao sao: bqcoAux
!       -rede de dados de observacao;
!       -modelo numerico;
!       -metodo de assimilacao de dados;
! Modelo numerico usado neste trabalho: 
!       -Modelo de agua rasa linear em duas dimensoes: as equacoes de agua rasa sao
!        derivadas dos principios de conservacao de massa e de momento. Essas equacoes
!        sao um conjunto de equacoes diferencias parcias  hiperbolicas derivadas das 
!        equacoes de Navier-Stokes.
!
! Para alterar as caracteristicas dos experimentos eh necessario alterar as seguintes 
! variaveis:
!       assimType: (1) Filtro de Kalman, (2) Rede neural artificial
!       freqObsX = freqObsY = 8 --> Experimento A
!       freqObsX = freqObsY = 5 --> Experimento B
!       timeStep =  60 --> Experimento A
!       timeStep = 100 --> Experimento B
!       qInitialCond = gaussiana --> Experimento A
!       uInitialCond = 0         --> Experimento A
!       vInitialCond = 0         --> Experimento A
!       qInitialCond = uInitialCond = vInitialCond = gaussiana --> Experimento B
!       neuronNumber = 10 --> numero neuronios na camada escondida 
!			      (definido empiricamente ou pelo MPCA)
! Quatro tipos de simulacoes com variacao do tamanho da grade:
! Simulacao 01: grade x=y=40
! Simulacao 02: grade x=y=100
! Simulacao 03: grade x=y=200
! Simulacao 04: grade x=y=1000
!
! Para gerar conjunto de dados para o treinamento, deve-se rodar esse programa com a 
! opcao, rodarah com filtro de Kalman
! assimType = 1
!
!
! Descricao das subrotinas
!       FUNCTION Droplet(h,w): simulate a water drop
!       SUBROUTINE model2d()
!       Globais.f90: contem as variaveis globais->
!****************************************************************************************
!****************************************************************************************
!****************************************************************************************
!****************************************************************************************
!
! Para rodar com RNA, alterar
! assimType = 2 e descomentar uma das linhas no codigo
!	yANN = qModelnorm
!	yANN = uModelnorm
!	yANN = vModelnorm
! e atualizar a variavel:
!	neuronNumber    = 10
!
! E descomentar uma das linhas
!	open(40, file='computingFKTime.out')
!	open(40, file='computingANNTime.out')
!
! Verificar para qual experimento vai ser rodado, olhar linhas acima!
!****************************************************************************************
!****************************************************************************************
!****************************************************************************************
!****************************************************************************************
USE KfsFunctions
!USE Globais

!****************************************************************************************
! Variables definition
!****************************************************************************************
IMPLICIT NONE

INTEGER :: i, j, k, s   !Laces index
INTEGER :: sX, sY, tS   !Index: step X, step Y, time Step
INTEGER :: dT
INTEGER :: hFluidMean
INTEGER :: freqObsX, freqObsY, freqObsT
INTEGER :: freqAssim
INTEGER :: gridX, gridY, timeStep
INTEGER :: dropStep
INTEGER :: dKalmanMatrix
INTEGER :: counterFreqAssim
INTEGER :: counterLine, counterCol
INTEGER :: assimType    !1-Kalman Filter, 2-ANN
INTEGER :: numPattern
INTEGER :: neuronNumber

REAL*8  :: percNoise
REAL*8  :: dX, dY
REAL*8  :: qDampCoeff, uDampCoeff, vDampCoeff
REAL*8  :: coriolis
REAL*8  :: gravityConst
REAL*8  :: dragCoeff
REAL*8  :: rhoAir
REAL*8  :: zonalW
REAL*8  :: rhoWatter
REAL*8  :: uExtForce, vExtForce
REAL*8  :: randNoise
REAL*8  :: initialAssimTime, endAssimTime, totalAssimTime
REAL*8  :: initialProcessTime, endProcessTime, totalProcessTime
REAL*8  :: a
REAL*8  :: valNormInf, valNormSup
!REAL*8  :: maxXann, minXann

!INTEGER, PARAMETER  :: X    = 10
!INTEGER, PARAMETER  :: Y    = 10
!DOUBLE PRECISION, DIMENSION(X,Y)      :: D
!DOUBLE PRECISION, DIMENSION(X,Y)      :: qGl, uGl, vGl !Nomes com terminacao Gl para enfatizar que eh Global
!DOUBLE PRECISION, DIMENSION(3*X*Y,3*X*Y)  :: matrixGl, matrixInvGl

DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:) :: D
DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:) :: qGl, uGl, vGl
DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:) :: matrixGl, matrixInvGl

DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:,:) :: qModelnorm
DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:,:) :: uModelnorm
DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:,:) :: vModelnorm
DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:,:) :: qObservnorm
DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:,:) :: uObservnorm
DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:,:) :: vObservnorm
DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:,:) :: qAnalysisnorm
DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:,:) :: uAnalysisnorm
DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:,:) :: vAnalysisnorm
DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:,:) :: qModel, uModel, vModel 
DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:,:) :: qObserv, uObserv, vObserv 
DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:,:) :: qAnalysis, uAnalysis, vAnalysis 
DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:,:) :: randNoiseObserv

DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:) :: qInitialCond, uInitialCond, vInitialCond
DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:) :: fFcast
DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:) :: pCovariance  
DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:) :: H
DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:) :: R 
DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:) :: pAnalysis
DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:) :: pFcast
DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:) :: kK
DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:) :: Q
DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:) :: vectorObserv
DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:) :: vectorModel
DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:) :: xAnalysis
DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:) :: identityMatrix
DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:) :: yFcast
DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:) :: auxMatMul, auxMatMul2
DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:) :: xANNNorm, xANN, transpQGl
DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:,:) :: yANN
DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:) :: wqco
DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:) :: bqcoAux, bqco
DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:) :: wqcs
DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:) :: bqcs
DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:,:) :: vco
DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:,:) :: vcs
DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:,:) :: yco
DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:,:,:) :: ycs

DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:) :: error

DOUBLE PRECISION :: qModelMax, qModelMin

INTEGER :: numThreads, tid, omp_get_num_threads, omp_get_thread_num
DOUBLE PRECISION :: omp_get_wtime

character(len=7) :: fnametemp
character(len=6) :: assimType_char, gridX_char, gridY_char, timeStep_char
character(len=6) :: freqObsT_char, freqObsX_char, freqObsY_char, percNoise_char
character(len=6) :: neuronNumber_char

CHARACTER(len=255) :: cmd
CALL get_command(cmd)
WRITE (*,*) TRIM(cmd)

print*,"command_argument_count(): ",command_argument_count()

 call get_command_argument(1,assimType_char)
 call get_command_argument(2,gridX_char)
 call get_command_argument(3,gridY_char)
 call get_command_argument(4,timeStep_char)
 call get_command_argument(5,freqObsT_char)
 call get_command_argument(6,freqObsX_char)
 call get_command_argument(7,freqObsY_char)
 call get_command_argument(8,percNoise_char)
 call get_command_argument(9,neuronNumber_char)

print*, "gridX: ", gridX_char
print*, "gridY: ", gridY_char
print*, "freqObsX: ", freqObsX_char
print*, "freqObsY: ", freqObsY_char
print*, "percNoise: ", percNoise_char
print*, "neuronNumber: ", neuronNumber_char 

assimType_char=trim(assimType_char)
gridX_char=trim(gridX_char)
gridY_char=trim(gridY_char)
timeStep_char=trim(timeStep_char)
freqObsT_char=trim(freqObsT_char)
freqObsX_char=trim(freqObsX_char)
freqObsY_char=trim(freqObsY_char)
percNoise_char=trim(percNoise_char)
neuronNumber_char=trim(neuronNumber_char)


fnametemp = 'temp.dat'
write (fnametemp,*) assimType_char
read (fnametemp,*) assimType
write (fnametemp,*) gridX_char
read (fnametemp,*) gridX
write (fnametemp,*) gridY_char
read (fnametemp,*) gridY
write (fnametemp,*) timeStep_char
read (fnametemp,*) timeStep
write (fnametemp,*) freqObsT_char
read (fnametemp,*) freqObsT
write (fnametemp,*) freqObsX_char
read (fnametemp,*) freqObsX
write (fnametemp,*) freqObsY_char
read (fnametemp,*) freqObsY
write (fnametemp,*) percNoise_char
read (fnametemp,*) percNoise
write (fnametemp,*) neuronNumber_char
read (fnametemp,*) neuronNumber


ALLOCATE(D(gridX,gridY))
ALLOCATE(qGl(gridX,gridY))
ALLOCATE(uGl(gridX,gridY))
ALLOCATE(vGl(gridX,gridY))
if (assimType .eq. 1) then 
   ALLOCATE(matrixGl(3*gridX*gridY,3*gridX*gridY))
   ALLOCATE(matrixInvGl(3*gridX*gridY,3*gridX*gridY))
endif

! Initialization of variables/parameters
dX = 100.0d3
dY = 100.0d3
dT = 180

qDampCoeff = 1.0/1.8d4
uDampCoeff = 1.0/1.8d4
vDampCoeff = 1.0/1.8d4

coriolis	= 1.0d-4
gravityConst= 9.8060
hFluidMean	= 5000
dragCoeff	= 1.6d-3
rhoAir		= 1.275
zonalW		= 5.0d0
rhoWatter	= 1.0d3

uExtForce = dragCoeff * rhoAir * zonalW * zonalW/(real(hFluidMean) * rhoWatter)
vExtForce = 0.0

!**************************************************************************************
! Alterar os valores do gridX e gridY:
! Simulacao 01: gridX = gridY = 40
! Simulacao 02: gridX = gridY = 100
! Simulacao 03: gridX = gridY = 200
! Simulacao 04: gridX = gridY = 1000
!gridX           = 10     !X -->Parametro global - Globais.f90
!gridY           = 10     !Y -->Parametro global - Globais.f90
!neuronNumber    = 8 
!**************************************************************************************
! Experimento A: 60 passos no tempo e 25 observacoes a cada 10 passos
!timeStep	= 60
!freqObsX	= 2
!freqObsY	= 2
!freqObsT	= 10

!**************************************************************************************
! Experimento B: 100 passos no tempo e 100 observacoes a cada 10 passos
!timeStep        = 100
!freqObsX        = 5
!freqObsY        = 5
!freqObsT        = 10

dKalmanMatrix = gridX * gridY

qGl = 0.0
uGl = 0.0
vGl = 0.0

!**************************************************************************************
! Assimilation type definition:
!	assimType = 1 -> Kalman filter
!	assimType = 2 -> ANN
!assimType = 1 

!**************************************************************************************
! Storing the time spent
initialProcessTime = omp_get_wtime()

if (assimType .eq. 1) then !Assimilacao com FK
   open(40, file='output/computingFKTime.out')
endif
if (assimType .eq. 2) then !Assimilacao com RNA
   open(40, file='output/computingANNTime.out')
endif


!**************************************************************************************
! Parametros usados quando assimType = 2 --> Rede Neural
numPattern = gridX * gridY
a = 1.0
! valores usados na normalizacao dos dados para a RNA
valNormInf = -1.0	 
valNormSup =  1.0

!**************************************************************************************
! Initial condition data
ALLOCATE(qInitialCond(gridX,gridY))
ALLOCATE(uInitialCond(gridX,gridY))
ALLOCATE(vInitialCond(gridX,gridY))
! Observation data
ALLOCATE(qObserv(gridX,gridY,timeStep))
ALLOCATE(uObserv(gridX,gridY,timeStep))
ALLOCATE(vObserv(gridX,gridY,timeStep))
! Model/Prediction data
ALLOCATE(qModel(gridX,gridY,timeStep))
ALLOCATE(uModel(gridX,gridY,timeStep))
ALLOCATE(vModel(gridX,gridY,timeStep))
! Analysis/Kalman Filter data
ALLOCATE(qAnalysis(gridX,gridY,timeStep))
ALLOCATE(uAnalysis(gridX,gridY,timeStep))
ALLOCATE(vAnalysis(gridX,gridY,timeStep))
ALLOCATE(randNoiseObserv(gridX,gridY,timeStep))
ALLOCATE(vectorModel(3*dKalmanMatrix,1))
if (assimType .eq. 1) then 
   ALLOCATE(vectorObserv(3*dKalmanMatrix,timeStep))
   !ALLOCATE(vectorModel(3*dKalmanMatrix,1))
   ALLOCATE(xAnalysis(3*dKalmanMatrix,1))
   ALLOCATE(yFcast(3*dKalmanMatrix,1))
   ALLOCATE(fFcast(3*dKalmanMatrix,3*dKalmanMatrix))
   ALLOCATE(pCovariance(3*dKalmanMatrix,3*dKalmanMatrix))
   ALLOCATE(pAnalysis(3*dKalmanMatrix,3*dKalmanMatrix))
   ALLOCATE(H(3*dKalmanMatrix,3*dKalmanMatrix))
   ALLOCATE(R(3*dKalmanMatrix,3*dKalmanMatrix))
   ALLOCATE(pFcast(3*dKalmanMatrix,3*dKalmanMatrix))
   ALLOCATE(kK(3*dKalmanMatrix,3*dKalmanMatrix))
   ALLOCATE(Q(3*dKalmanMatrix,3*dKalmanMatrix))
   ALLOCATE(identityMatrix(3*dKalmanMatrix,3*dKalmanMatrix))
   ALLOCATE(auxMatMul(3*dKalmanMatrix,3*dKalmanMatrix))
   ALLOCATE(auxMatMul2(3*dKalmanMatrix,3*dKalmanMatrix))
endif
ALLOCATE(error(timeStep))

ALLOCATE(yANN(1,gridX*gridY,timeStep))
ALLOCATE(xANN(2, gridX*gridY))
ALLOCATE(xANNNorm(2, gridX*gridY))
ALLOCATE(transpQGl(gridX, gridY))

ALLOCATE(wqco(neuronNumber,2))
ALLOCATE(bqcoAux(1,neuronNumber))
ALLOCATE(bqco(neuronNumber,1))
ALLOCATE(wqcs(1,neuronNumber))
ALLOCATE(bqcs(1,1))

!$OMP PARALLEL
numThreads = omp_get_num_threads()
!print*, "omp_get_thread_num(): ", omp_get_thread_num()
!$OMP END PARALLEL

print*
print*, "numThreads: ", numThreads
print*

ALLOCATE(vco(neuronNumber,1,numThreads))
ALLOCATE(vcs(1,1,numThreads))
ALLOCATE(yco(neuronNumber,1,numThreads))
ALLOCATE(ycs(1,1,numThreads))

!Normalization
ALLOCATE(qObservnorm(gridX,gridY,timeStep))
ALLOCATE(uObservnorm(gridX,gridY,timeStep))
ALLOCATE(vObservnorm(gridX,gridY,timeStep))
ALLOCATE(qModelnorm(gridX,gridY,timeStep))
ALLOCATE(uModelnorm(gridX,gridY,timeStep))
ALLOCATE(vModelnorm(gridX,gridY,timeStep))
ALLOCATE(qAnalysisnorm(gridX,gridY,timeStep))
ALLOCATE(uAnalysisnorm(gridX,gridY,timeStep))
ALLOCATE(vAnalysisnorm(gridX,gridY,timeStep))

!**************************************************************************************
if (assimType .eq. 2) then !Assimilacao com RNA
	!Reading the weights files and bias for Neural Network
	open(10, file = './data/wqcoExpA.dat')
	do j = 1, 2
    		read(10,*)(wqco(i,j),i = 1,neuronNumber)
	enddo
	close(10)

	open(10, file = './data/bqcoExpA.dat')
	    read(10,*)(bqco(i,1),i = 1, neuronNumber)
	close(10)

	open(10, file = './data/wqcsExpA.dat')
	    read(10,*)(wqcs(1,j),j = 1,neuronNumber)
	close(10)

	open(10, file = './data/bqcsExpA.dat')
	    read(10,*)bqcs(1,1)
	close(10)
endif


!wqco = 0.5
!bqco = 0.5
!wqcs = 0.5
!bqcs = 0.5


qObserv = 0.0
uObserv = 0.0
vObserv = 0.0

!**************************************************************************************
! Initial Condition
dropStep = 5
call droplet(dropStep,gridX,D) ! Simulate a water drop

!**************************************************************************************
!Foi comentado este trecho para validar o teste com matlab,
!lah tambem foi usado o mesmo valor aleatorio = 0.7921
!call srand(0)
!aleat = rand()
randNoise = 0.7921

!**************************************************************************************
! Experimento A --> condicao incial q=gaussiana + ruido; u=v=0
!qInitialCond = qInitialCond + randNoise * D * (sqrt(2.0))
!uInitialCond = uInitialCond + randNoise * (sqrt(2.0))
!vInitialCond = vInitialCond + randNoise * (sqtr(2.0))

!**************************************************************************************
! Experimento B --> condicao incial q=u=v=gaussiana + ruido
qInitialCond = qInitialCond + randNoise * D * (sqrt(2.0))
uInitialCond = uInitialCond + randNoise * D * (sqrt(2.0))
vInitialCond = vInitialCond + randNoise * D * (sqrt(2.0))

open(10, file = 'output/qInitialCondExpA.out')
!open(11, file = 'output/uInitialCondExpA.out')
!open(12, file = 'output/vInitialCondExpA.out')
do sX = 1, gridX
    do sY = 1, gridY
        write(10,*)qInitialCond(sX, sY)
        !write(11,*)uInitialCond(sX, sY)
        !write(12,*)vInitialCond(sX, sY)
    enddo
enddo
close(10)
!close(11)
!close(12)
print*,'SALVOU CONDICAO INICIAL - qInitialCondExpA.out'

qGl = qInitialCond
uGl = uInitialCond
vGl = vInitialCond

!**************************************************************************************
!Integrando o modelo no tempo
do tS = 1, timeStep
    call model2d(dX, dY, dT, gridX, gridY, hFluidMean, qDampCoeff, uDampCoeff, vDampCoeff, coriolis, gravityConst, qGl, uGl, vGl)
    qModel(:,:,tS) = qGl
    uModel(:,:,tS) = uGl
    vModel(:,:,tS) = vGl
enddo

if (assimType .eq. 1) then
!Escrevendo dados em todo o dominio 2D, e todos os timesteps:
open(10, file = 'output/full/qModelExpA.out')
!open(11, file = 'output/full/uModelExpA.out')
!open(12, file = 'output/full/vModelExpA.out')
do tS = 1, timeStep
    do sX = 1, gridX
        do sY = 1, gridY
            write(10,'(6X,F10.6)',advance='no') qModel(sX, sY, tS)
            !write(11,'(6X,F8.5)',advance='no') uModel(sX, sY, tS)
            !write(12,'(6X,F8.5)',advance='no') vModel(sX, sY, tS)
        enddo
    enddo
enddo
close(10)
!close(11)
!close(12)
print*,'SALVOU RESULTADO DA INTEGRACAO DO MODELO - qModelExpA.out'
endif

initialTime = omp_get_wtime()
call srand(0)
do tS = 1, timeStep
   do sY = 1, gridY
      do sX = 1, gridX
         randNoise = 2*rand()-1
         !randNoiseObserv(sX,sY,tS) = randNoise * sqrt(0.01)
         qObserv(sX,sY,tS) = qModel(sX,sY,tS) + percNoise*qModel(sX,sY,tS)*randNoise 
         uObserv(sX,sY,tS) = uModel(sX,sY,tS) + percNoise*uModel(sX,sY,tS)*randNoise 
         vObserv(sX,sY,tS) = vModel(sX,sY,tS) + percNoise*vModel(sX,sY,tS)*randNoise 
      enddo
   enddo
enddo
endTime = omp_get_wtime()
print*,'Tempo de insercao do ruido  : ', endTime-initialTime
print*,'Gerou o ruido que sera adicionado ao modelo - gerando as observacoes '

!qObserv = qModel + randNoiseObserv
!uObserv = uModel + randNoiseObserv
!vObserv = vModel + randNoiseObserv

if (assimType .eq. 1) then
!Escrevendo dados em todo o dominio 2D, e todos os timesteps:
open(10, file = 'output/full/qObservExpA.out')
!open(11, file = 'output/full/uObservExpA.out')
!open(12, file = 'output/full/vObservExpA.out')
do tS = 1, timeStep
    do sX = 1, gridX
        do sY = 1, gridY
            write(10,'(6X,F10.6)',advance='no') qObserv(sX, sY, tS)
            !write(11,'(6X,F8.5)',advance='no') uObserv(sX, sY, tS)
            !write(12,'(6X,F8.5)',advance='no') vObserv(sX, sY, tS)
        enddo
    enddo
enddo
close(10)
!close(11)
!close(12)
print*,'SALVOU AS OBSERVACOES -- MODELO + Rand - qObservExpA.out'
endif

qGl = qInitialCond
uGl = uInitialCond
vGl = vInitialCond

if (assimType .eq. 1) then 
!**************************************************************************************
! Generating observation vector - y
do tS = freqObsT, timeStep, freqObsT
    s = 1
    do sX = 1, gridX
        vectorObserv(s:s+gridY-1, tS) = qObserv(sX,:,tS)
        s = s + gridY
    enddo
    do sX = 1, gridX
        vectorObserv(s:s+gridY-1, tS) = uObserv(sX,:,tS)
        s = s + gridY
    enddo
    do sX = 1, gridX
        vectorObserv(s:s+gridY-1, tS) = vObserv(sX,:,tS)
        s = s + gridY
    enddo
enddo

!**************************************************************************************
! Kalman Filter Assimilation
print*, 'Inicializou a freqAssim'
auxMatMul   = 0.0
print*, 'Inicializando as matrizes 1'
auxMatMul2  = 0.0
print*, 'Inicializando as matrizes 2'
matrixGl    = 0.0
print*, 'Inicializando as matrizes 3'
matrixInvGl = 0.0
print*, 'Inicializando as matrizes 4'
kK          = 0.0
print*, 'Inicializando as matrizes 5'
H           = 0.0
print*, 'Inicializando as matrizes 6'
fFcast      = 0.0
print*, 'Inicializando as matrizes 7'
pCovariance = 0.0
print*, 'Inicializando as matrizes 8'
R           = 0.0
print*, 'Inicializando as matrizes 9'
Q           = 0.0
print*, 'Inicializando as matrizes 10'
pFcast      = 0.0

print*,'Completando as matrizes com valores apropriados'
do sY = 1, 3*dKalmanMatrix
    do sX = 1, 3*dKalmanMatrix
        fFcast(sX,sY) = 0.0         !Funcao das variaveis de estado do modelo
        pCovariance(sX,sY) = 0.0    !Matriz de covariancia do erro de previsao
        H(sX,sY) = 0.0              !Matriz de observacao
        R(sX,sY) = 0.0              !Matriz de covariancia do erro de observacao
        Q(sX,sY) = 0.0              !Matriz de covariancia do erro de modelagem
        pFcast(sX,sY) = 0.0         !Matriz de covariancia do erro de previsao
        kK(sX,sY) = 0.0             !Ganho de Kalman
    enddo
    pCovariance(sY,sY)= 1.0
    H(sY,sY) = 1.0
    R(sY,sY) = 1.0
    Q(sY,sY) = 0.003
enddo

pAnalysis = pCovariance             !Matriz de covariancia da analise
endif

freqAssim = freqObsT

qModelMax = maxval(qModel)
qModelMin = minval(qModel)
!**************************************************************************************
! Processo de normalizacao dos dados para serem usados na RNA

if (assimType .EQ. 2) then
    print*, 'Normalizando os dados para RNA'
    qModelnorm = (maxval(qModel) * valNormInf - minval(qModel) * valNormSup + qModel * &
		&(valNormSup - valNormInf)) / (maxval(qModel) - minval(qModel))
    qObservnorm = (maxval(qObserv) * valNormInf - minval(qObserv) * valNormSup + qObserv * &
		&(valNormSup - valNormInf)) / (maxval(qObserv) - minval(qObserv))
    uModelnorm = (maxval(uModel) * valNormInf - minval(uModel) * valNormSup + uModel * &
                &(valNormSup - valNormInf)) / (maxval(uModel) - minval(uModel))
    uObservnorm = (maxval(uObserv) * valNormInf - minval(uObserv) * valNormSup + uObserv * &
                &(valNormSup - valNormInf)) / (maxval(uObserv) - minval(uObserv))
    vModelnorm = (maxval(vModel) * valNormInf - minval(vModel) * valNormSup + vModel * &
                &(valNormSup - valNormInf)) / (maxval(vModel) - minval(vModel))
    vObservnorm = (maxval(vObserv) * valNormInf - minval(vObserv) * valNormSup + vObserv * &
                &(valNormSup - valNormInf)) / (maxval(vObserv) - minval(vObserv))
endif

!**************************************************************************************
! Inicializacao da variavel yANN para ser usada na RNA
! tirar o comentario da linha que se deseja avaliar
yANN = qModelnorm
!yANN = uModelnorm
!yANN = vModelnorm

 counterFreqAssim = 0
 totalAssimTime = 0.0d+00

do tS = 1, timeStep
    call model2d(dX, dY, dT, gridX, gridY, hFluidMean, qDampCoeff, uDampCoeff, vDampCoeff, coriolis, gravityConst, qGl, uGl, vGl)
    qAnalysis(:,:,tS) = qGl
    uAnalysis(:,:,tS) = uGl
    vAnalysis(:,:,tS) = vGl
    ! generating state variable vector - x
    s = 1
    do sX = 1, gridX
        vectorModel(s:s+gridY-1,1) = qGl(sX,:)
        s = s + gridY
    enddo
    do sX = 1, gridX
        vectorModel(s:s+gridY-1,1) = uGl(sX,:)
        s = s + gridY
    enddo
    do sX = 1, gridX
        vectorModel(s:s+gridY-1,1) = vGl(sX,:)
        s = s + gridY
    enddo

    counterFreqAssim = counterFreqAssim + 1

    if (counterFreqAssim .EQ. freqAssim)  then
        SELECT CASE (assimType)
        CASE (1)
	    !**************************************************************************
            call CPU_TIME(initialAssimTime)
            print*, 'FK Assimilation cycle - timeStep', tS
            counterFreqAssim = 0
            do i = 1, dKalmanMatrix - 1
                !Elementos da diagonal principal da matriz
                fFcast(i, i) = 1 - dT * qDampCoeff
                fFcast(i + dKalmanMatrix, i + dKalmanMatrix) = 1 + dT*uDampCoeff
                fFcast(i + 2*dKalmanMatrix, i + 2*dKalmanMatrix) = 1 + dT*vDampCoeff
                !Elementos acima da diagonal principal
                fFcast(i, i + dKalmanMatrix) = dT*hFluidMean/dX
                fFcast(i + dKalmanMatrix, i + 2*dKalmanMatrix) = dT*coriolis/4
                fFcast(i, i + dKalmanMatrix + gridY) = -dT*hFluidMean/dX
                fFcast(i + dKalmanMatrix, i + 2*dKalmanMatrix + 1) =  dT*coriolis/4
                fFcast(i, i + 2*dKalmanMatrix) = dT * hFluidMean/dY
                fFcast(i, i + 2*dKalmanMatrix + 1) = -dT*hFluidMean/dY
                !Elementos abaixo da diagonal principal
                fFcast(i + dKalmanMatrix, i) = -dT*gravityConst/dX+1
                fFcast(i + 2*dKalmanMatrix, i + dKalmanMatrix) = -dT*coriolis/4+1
                fFcast(i + 2*dKalmanMatrix, i) =  dT*gravityConst/dY + 1
                fFcast(i + dKalmanMatrix + gridX, i) =  dT*gravityConst/dX+1
                fFcast(i + 2*dKalmanMatrix, i + dKalmanMatrix + gridY) =  -dT*coriolis/4+1
                fFcast(i + 2*dKalmanMatrix, i + dKalmanMatrix + gridY + 1) =  -dT*coriolis/4+1
                fFcast(i + 2*dKalmanMatrix + 1, i) = dT * gravityConst/dY+1
                fFcast(i + 2*dKalmanMatrix + 1, i + dKalmanMatrix)  = -dT*coriolis/4+1
            enddo
            fFcast(gridX*3 + 1, dKalmanMatrix + 1) = -dT*hFluidMean/dX
            fFcast(gridX*3 + 2, dKalmanMatrix + 2) = -dT*hFluidMean/dX
            fFcast(gridX*3 + 3, dKalmanMatrix + 3) = -dT*hFluidMean/dX
            fFcast(3*dKalmanMatrix, dKalmanMatrix + 1) = dT*gravityConst/dY
            fFcast(3*dKalmanMatrix, dKalmanMatrix) = -dT*gravityConst/dY
            fFcast(3*dKalmanMatrix, dKalmanMatrix + 3) = -dT*coriolis/4
            fFcast(3*dKalmanMatrix, dKalmanMatrix + 4) = -dT*coriolis/4
            fFcast(3*dKalmanMatrix, 2*dKalmanMatrix - 1) = -dT*coriolis/4
            fFcast(3*dKalmanMatrix, 2*dKalmanMatrix) = -dT*coriolis/4
            fFcast(2*dKalmanMatrix, 3*dKalmanMatrix) = -dT*coriolis/4
            fFcast(3*dKalmanMatrix, 3*dKalmanMatrix) = (1 + dT*vDampCoeff)
            fFcast(dKalmanMatrix + 1, 3*dKalmanMatrix - (gridY + 3)) = dT*coriolis/4
            fFcast(dKalmanMatrix + 2, 3*dKalmanMatrix - (gridY + 2)) = dT*coriolis/4
            fFcast(dKalmanMatrix + 3, 3*dKalmanMatrix - (gridY + 1)) = dT*coriolis/4
            fFcast(dKalmanMatrix + 1, 3*dKalmanMatrix - (gridY + 2)) = dT*coriolis/4
            fFcast(dKalmanMatrix + 2, 3*dKalmanMatrix - (gridY + 1)) = dT*coriolis/4
            fFcast(dKalmanMatrix + 3, 3*dKalmanMatrix - gridY) = dT*coriolis/4
            fFcast(dKalmanMatrix + 1, 2*gridY + 1) = dT*gravityConst/dX
            fFcast(dKalmanMatrix + 2, 2*gridY + 2) = dT*gravityConst/dX
            fFcast(dKalmanMatrix + 3, 2*gridY + 3) = dT*gravityConst/dX
            fFcast(dKalmanMatrix + 4, 2*gridY + 4) = dT*gravityConst/dX
            fFcast(dKalmanMatrix, 3*dKalmanMatrix - gridY) = dT*hFluidMean/dY
            fFcast = fFcast(1:3*dKalmanMatrix, 1:3*dKalmanMatrix)
            ! Error covariance forecast (Furtado, 2011, pg 41, equation 3.60)
            print*, 'Primeira chamada dgemm'
            !dgemm('normal', 'normal',nLinMatrizA,nColMatrizB,tamanhoMatrizC,...
            call dgemm('n','n',3*dKalmanMatrix,3*dKalmanMatrix,3*dKalmanMatrix,1.0d0,fFcast,3*dKalmanMatrix,&
                pAnalysis,3*dKalmanMatrix,0.0d0,auxMatMul,3*dKalmanMatrix)
            !!pFcast = matmul(auxMatMul,(transpose(fFcast))) + Q
            call dgemm('n','t',3*dKalmanMatrix,3*dKalmanMatrix,3*dKalmanMatrix,1.0d0,auxMatMul,3*dKalmanMatrix,&
                fFcast,3*dKalmanMatrix,0.0d0,auxMatMul2,3*dKalmanMatrix)
            pFcast = auxMatMul2 + Q
            print*, 'Passou pela equacao 3.60'
            ! Kalman gain computation (Furtado, 2011, pg 41, equation 3.61)
            !auxMatMul = matmul(H, pFcast)
            print*, 'Segunda chamada dgemm'
            call dgemm('n','n',3*dKalmanMatrix,3*dKalmanMatrix,3*dKalmanMatrix,1.0d0,H,3*dKalmanMatrix,&
                pFcast, 3*dKalmanMatrix,0.0d0,auxMatMul,3*dKalmanMatrix)
            !matriz = matmul(auxMatMul,transpose(H)) + R
            print*, 'Terceira chamada dgemm'
            call dgemm('n','t',3*dKalmanMatrix,3*dKalmanMatrix,3*dKalmanMatrix,1.0d0,auxMatMul,3*dKalmanMatrix,&
                H,3*dKalmanMatrix,0.0d0,matrixGl,3*dKalmanMatrix)
            print*,'Invertendo matriz'
            matrixInvGl = inv(matrixGl)
            !auxMatMul = matmul(pFcast, transpose(H))
            print*,'Quarta chamada dgemm'
            call dgemm('n','t',3*dKalmanMatrix,3*dKalmanMatrix,3*dKalmanMatrix,1.0d0,pFcast,3*dKalmanMatrix,&
                H,3*dKalmanMatrix,0.0d0,auxMatMul,3*dKalmanMatrix)
            print*,'Quinta chamada dgemm'
            !kK = matmul(auxMatMul, MatrizInv)
            call dgemm('n','n',3*dKalmanMatrix,3*dKalmanMatrix,3*dKalmanMatrix,1.0d0,auxMatMul,3*dKalmanMatrix,&
                matrixInvGl,3*dKalmanMatrix,0.0d0,kK,3*dKalmanMatrix)
            print*, 'Passou pela equacao 3.61'
            ! Calculo da estimativa (Furtado, 2011, pg 41, equation 3.62)
            yFcast = matmul(H,vectorModel)
            !call dgemm('n','n',3*dKalmanMatrix,3*dKalmanMatrix,3*dKalmanMatrix,1.0d0,H,3*dKalmanMatrix,&
            !vectorModel,3*dKalmanMatrix,0.0d0,yFcast,3*dKalmanMatrix)
            print*, 'Passou pela equacao 3.62'
            ! State analysis (Furtado, 2011, pg 41, equation 3.63)
            xAnalysis(:,1) = vectorModel(:,1) + matmul(kK(:,:), (vectorObserv(:,tS) - yFcast(:,1)))
            print*, 'Passou pela equacao 3.63'
            do i = 1, 3*dKalmanMatrix
                if (vectorObserv(i,tS) .EQ. 0) then
                    xAnalysis(i,1) = vectorModel(i,1)
                endif
            enddo
            do i = 1 , 3*dKalmanMatrix
                do j = 1 , 3*dKalmanMatrix
                    if ( i /= j ) then
                        identityMatrix(i, j) = 0.0
                    else
                        identityMatrix(i, j) = 1.0
                    endif
                enddo
            enddo
            ! Error covariance of analysis (Furtado, 2011, pg 41, equation 3.64)
            !auxMatMul = matmul(kK,H)
            call dgemm('n','n',3*dKalmanMatrix,3*dKalmanMatrix,3*dKalmanMatrix,1.0d0,kK,3*dKalmanMatrix,&
                H,3*dKalmanMatrix,0.0d0,auxMatMul,3*dKalmanMatrix)
            !pAnalysis = matmul(identityMatrix - auxMatMul, pFcast)
            auxMatMul2 = identityMatrix - auxMatMul
            call dgemm('n','t',3*dKalmanMatrix,3*dKalmanMatrix,3*dKalmanMatrix,1.0d0,auxMatMul2,3*dKalmanMatrix,&
                pFcast,3*dKalmanMatrix,0.0d0,pAnalysis,3*dKalmanMatrix)
            print*,'passou pela equacao 3.64'
            counterLine = 1
            do i = 1, 3
                do j = 1, gridX
                    do k = 1, gridY
                        if (i .EQ. 1) then
                            qGl(j, k) = xAnalysis(counterLine,1)
                        else if (i .EQ. 2) then
                            uGl(j, k) = xAnalysis(counterLine,1)
                        else if (i .EQ. 3) then
                            vGl(j, k) = xAnalysis(counterLine,1)
                        endif
                        counterLine = counterLine + 1
                    enddo
                enddo
            enddo

            qAnalysis(:,:,tS) = qGl
            uAnalysis(:,:,tS) = uGl
            vAnalysis(:,:,tS) = vGl


            call CPU_TIME(endAssimTime)
            totalAssimTime = endAssimTime - initialAssimTime
            print*,'FK Assimilation time: ', totalAssimTime, tS
            write(40,*) 'FK Assimilation time: ', totalAssimTime, tS

        CASE (2)
	    !**************************************************************************

	    print*, 'ANN Assimilation cycle - timeStep', tS
        counterFreqAssim = 0

	    counterCol = 1
            do sX = 1, gridX
                do sY = 1, gridY
                    xANN(1,counterCol) = qModelnorm(sX, sY,tS)
                    xANN(2,counterCol) = qObservnorm(sX,sY,tS)
                    !xANN(1,counterCol) = uModelnorm(sX, sY,tS)
                    !xANN(2,counterCol) = uObservnorm(sX,sY,tS)
                    !xANN(1,counterCol) = vModelnorm(sX, sY,tS)
                    !xANN(2,counterCol) = vObservnorm(sX,sY,tS)
                    counterCol = counterCol + 1
                enddo
            enddo

	        !print*,'TUDO PRONTO PARA A RNA'

            initialAssimTime = omp_get_wtime()
!$OMP PARALLEL DO         &
!$OMP DEFAULT(shared)     &
!$OMP PRIVATE(sX,sY,i,tid)                 
            do sX = 1, gridX
                do sY = 1, gridY
                   tid = omp_get_thread_num() + 1
                   i = (sX-1)*gridY + sY
                   vco(:,1,tid) = matmul(wqco(:,:),xANN(:,i))
                   vco(:,1,tid) = vco(:,1,tid) - (bqco(:,1))
                   yco(:,1,tid) = (1.d0 - DEXP(-vco(:,1,tid))) / (1.d0 + DEXP(-vco(:,1,tid)))
                   vcs(:,1,tid) = matmul(wqcs(:,:), yco(:,1,tid))
                   vcs(:,1,tid) = vcs(:,1,tid) - bqcs(:,1)
                   yANN(:,i,tS) = (1.d0-DEXP(-vcs(:,1,tid)))/(1.d0+DEXP(-vcs(:,1,tid)))
                   qGl(sX,sY) = (yANN(1,i,tS)*(qModelMax-qModelMin) + qModelMax + qModelMin)/2.0 
                enddo
            enddo
!$OMP END PARALLEL DO
            endAssimTime = omp_get_wtime()
            totalAssimTime = totalAssimTime + (endAssimTime - initialAssimTime)
           
            qAnalysis(:,:,tS) = qGl

            
            !print*,'PASSAMOS PELA RNA'

        CASE(3)
          print*,'Chamar FPGA'

        END SELECT

    endif
enddo

print*,'ANN Assimilation time: ', totalAssimTime, tS
write(40,*)'ANN Assimilation time: ', totalAssimTime, tS


if (assimType .eq. 3) then !Processo de desnormalizacao da saida de RNA
!	qAnalysis = (yANN * (maxval(qModel) - minval(qModel)) - maxval(qModel) * valNormInf +&
!			& minval(qModel) * valNormSup) / (valNormSup - valNormInf)
!Escrevendo dados em todo o dominio 2D, e todos os timesteps:
open(10, file = 'output/full/qAnalysisExpA_RNA.out')
!open(11, file = 'output/full/uAnalysisExpA.out')
!open(12, file = 'output/full/vAnalysisExpA.out')
do tS = 1, timeStep
    do sX = 1, gridX
        do sY = 1, gridY
            write(10,'(6X,F10.6)',advance='no') qAnalysis(sX, sY, tS)
	    !write(11,'(6X,F8.5)',advance='no') uAnalysis(sX, sY, tS)
	    !write(12,'(6X,F8.5)',advance='no') vAnalysis(sX, sY, tS)
        enddo
    enddo
enddo
close(10)
!close(11)
!close(12)		
endif

if (assimType .eq. 3) then 
!Escrevendo dados em todo o dominio 2D, e todos os timesteps:
open(10, file = 'output/full/qAnalysisExpA.out')
!open(11, file = 'output/full/uAnalysisExpA.out')
!open(12, file = 'output/full/vAnalysisExpA.out')
do tS = 1, timeStep
    do sX = 1, gridX
        do sY = 1, gridY
            write(10,'(6X,F10.6)',advance='no') qAnalysis(sX, sY, tS)
	    !write(11,'(6X,F8.5)',advance='no') uAnalysis(sX, sY, tS)
	    !write(12,'(6X,F8.5)',advance='no') vAnalysis(sX, sY, tS)
        enddo
    enddo
enddo
close(10)
!close(11)
!close(12)


!qAnalysisnorm = (maxval(qAnalysis) * valNormInf - minval(qAnalysis) * valNormSup + qAnalysis * &
!	&(valNormSup - valNormInf)) / (maxval(qAnalysis) - minval(qAnalysis))
!Escrevendo dados a serem utilizados no MPCA
open(10, file = 'output/training/qAnalysisExpA.out')
!open(11, file = 'output/training/uAnalysisExpA.out')
!open(12, file = 'output/training/vAnalysisExpA.out')
do tS = freqObsT, timeStep, freqObsT
    do sX = freqObsX, gridX, freqObsX
        do sY = freqObsY, gridY, freqObsY
            write(10,'(6X,F10.6)',advance='no') qAnalysis(sX, sY, tS)
	    !write(11,'(6X,F8.5)',advance='no') uAnalysis(sX, sY, tS)
	    !write(12,'(6X,F8.5)',advance='no') vAnalysis(sX, sY, tS)
        enddo
    enddo
enddo
close(10)
!close(11)
!close(12)

!qObservnorm = (maxval(qObserv) * valNormInf - minval(qObserv) * valNormSup + qObserv * &
!	&(valNormSup - valNormInf)) / (maxval(qObserv) - minval(qObserv))
!Escrevendo dados a serem utilizados no MPCA
open(10, file = 'output/training/qObservExpA.out')
!open(11, file = 'output/training/uObservExpA.out')
!open(12, file = 'output/training/vObservExpA.out')
do tS = freqObsT, timeStep, freqObsT
    do sX = freqObsX, gridX, freqObsX
        do sY = freqObsY, gridY, freqObsY
            write(10,'(6X,F10.6)',advance='no') qObserv(sX, sY, tS)
            !write(11,'(6X,F8.5)',advance='no') uObserv(sX, sY, tS)
            !write(12,'(6X,F8.5)',advance='no') vObserv(sX, sY, tS)
        enddo
    enddo
enddo
close(10)
!close(11)
!close(12)

!qModelnorm = (maxval(qModel) * valNormInf - minval(qModel) * valNormSup + qModel * &
!	&(valNormSup - valNormInf)) / (maxval(qModel) - minval(qModel))
!Escrevendo dados a serem utilizados no MPCA
open(10, file = 'output/training/qModelExpA.out')
!open(11, file = 'output/training/uModelExpA.out')
!open(12, file = 'output/training/vModelExpA.out')
do tS = freqObsT, timeStep, freqObsT
    do sX = freqObsX, gridX, freqObsX
        do sY = freqObsY, gridY, freqObsY
            write(10,'(6X,F10.6)',advance='no') qModel(sX, sY, tS)
            !write(11,'(6X,F8.5)',advance='no') uModel(sX, sY, tS)
            !write(12,'(6X,F8.5)',advance='no') vModel(sX, sY, tS)
        enddo
    enddo
enddo
close(10)
!close(11)
!close(12)

endif

endProcessTime = omp_get_wtime()

totalProcessTime = endProcessTime - initialProcessTime
print*,'Total Process time: ', totalProcessTime
write(40,*) 'Total Process time:', totalProcessTime

close(40)

print*,'FIM'

! Initial condition data
if (allocated(qInitialCond)) DEALLOCATE(qInitialCond)
if (allocated(uInitialCond)) DEALLOCATE(uInitialCond)
if (allocated(vInitialCond)) DEALLOCATE(vInitialCond)
! Observation data
if (allocated(qObserv)) DEALLOCATE(qObserv)
if (allocated(uObserv)) DEALLOCATE(uObserv)
if (allocated(vObserv)) DEALLOCATE(vObserv)
! Model/Prediction data
if (allocated(qModel)) DEALLOCATE(qModel)
if (allocated(uModel)) DEALLOCATE(uModel)
if (allocated(vModel)) DEALLOCATE(vModel)
! Analysis/Kalman Filter data
if (allocated(qAnalysis)) DEALLOCATE(qAnalysis)
if (allocated(uAnalysis)) DEALLOCATE(uAnalysis)
if (allocated(vAnalysis)) DEALLOCATE(vAnalysis)

if (allocated(randNoiseObserv)) DEALLOCATE(randNoiseObserv)
if (allocated(vectorObserv)) DEALLOCATE(vectorObserv)
if (allocated(vectorModel)) DEALLOCATE(vectorModel)
if (allocated(xAnalysis)) DEALLOCATE(xAnalysis)
if (allocated(yFcast)) DEALLOCATE(yFcast)
if (allocated(fFcast)) DEALLOCATE(fFcast)
if (allocated(pCovariance)) DEALLOCATE(pCovariance)
if (allocated(pAnalysis)) DEALLOCATE(pAnalysis)
if (allocated(H)) DEALLOCATE(H)
if (allocated(R)) DEALLOCATE(R)
if (allocated(pFcast)) DEALLOCATE(pFcast)
if (allocated(kK)) DEALLOCATE(kK)
if (allocated(Q)) DEALLOCATE(Q)
if (allocated(identityMatrix)) DEALLOCATE(identityMatrix)
if (allocated(error)) DEALLOCATE(error)
if (allocated(wqco)) DEALLOCATE(wqco)
if (allocated(bqcoAux)) DEALLOCATE(bqcoAux)
if (allocated(bqco)) DEALLOCATE(bqco)
if (allocated(wqcs)) DEALLOCATE(wqcs)
if (allocated(bqcs)) DEALLOCATE(bqcs)
if (allocated(vco)) DEALLOCATE(vco)
if (allocated(vcs)) DEALLOCATE(vcs)
if (allocated(yco)) DEALLOCATE(yco)
if (allocated(ycs)) DEALLOCATE(ycs)
if (allocated(yANN)) DEALLOCATE(yANN)
if (allocated(xANN)) DEALLOCATE(xANN)
if (allocated(xANNNorm)) DEALLOCATE(xANNNorm)

if (allocated(qModelnorm)) DEALLOCATE(qModelnorm)
if (allocated(uModelnorm)) DEALLOCATE(uModelnorm)
if (allocated(vModelnorm)) DEALLOCATE(vModelnorm)
if (allocated(qObservnorm)) DEALLOCATE(qObservnorm)
if (allocated(uObservnorm)) DEALLOCATE(uObservnorm)
if (allocated(vObservnorm)) DEALLOCATE(vObservnorm)
if (allocated(qAnalysisnorm)) DEALLOCATE(qAnalysisnorm)
if (allocated(uAnalysisnorm)) DEALLOCATE(uAnalysisnorm)
if (allocated(vAnalysisnorm)) DEALLOCATE(vAnalysisnorm)

if (allocated(D)) DEALLOCATE(D)
if (allocated(qGl)) DEALLOCATE(qGl)
if (allocated(uGl)) DEALLOCATE(uGl)
if (allocated(vGl)) DEALLOCATE(vGl)
if (allocated(matrixGl)) DEALLOCATE(matrixGl)
if (allocated(matrixInvGl)) DEALLOCATE(matrixInvGl)

if (allocated(auxMatMul)) DEALLOCATE(auxMatMul)
if (allocated(auxMatMul2)) DEALLOCATE(auxMatMul2)
if (allocated(transpQGl)) DEALLOCATE(transpQGl)

END PROGRAM
