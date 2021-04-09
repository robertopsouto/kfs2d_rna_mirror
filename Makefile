#Compilador
#FC = /usr/local/bin/gfortran
FC = gfortran
#Opcoes de compilacao
FFLAGS = -c -O2

#Opcoes de compilacao
FFLAGSOPT = -O2

#Bibliotecas
FFLIBS = -llapack -lblas -g -fcheck=all -Wall -fbacktrace

#Objetos
objects = kfsFunctions.o KFS2d.o

KFS2d:$(objects)
	$(FC) -o ./KFS2d $(objects) $(FFLIBS)

KFS2d.o:
	gfortran -O2 -c -g ./src/KFS2d.f90

kfsFunctions.o:
	gfortran -O2 -c -g ./src/kfsFunctions.f90

clean:
	rm -rf ./src/*.mod *.o KFS2d *.mod *.o *.out
