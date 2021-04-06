MODULE Globais
IMPLICIT NONE

    INTEGER, PARAMETER  :: X    = 10
    INTEGER, PARAMETER  :: Y    = 10
    DOUBLE PRECISION, DIMENSION(X,Y)      :: D
    DOUBLE PRECISION, DIMENSION(X,Y)      :: qGl, uGl, vGl !Nomes com terminacao Gl para enfatizar que eh Global
    DOUBLE PRECISION, DIMENSION(3*X*Y,3*X*Y)  :: matrixGl, matrixInvGl


END MODULE Globais
