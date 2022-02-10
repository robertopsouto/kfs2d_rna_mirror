!***********************************************************************
! Traducao da funcao ndgrid do Matlab para o Fortran
! Tradutora: Sabrina Sambati (CAP/INPE)
!***********************************************************************

MODULE KfsFunctions
!    use Globais

CONTAINS

    !***********************************************************************
    ! Traducao da funcao droplet e ndgrid do Matlab para o Fortran
    !SUBROUTINE droplet(height,width,Dd)
    !***********************************************************************
    SUBROUTINE droplet(height,width,D)
        IMPLICIT NONE
        integer, intent(in) :: height, width
        double precision, intent(inout) :: D(:,:)
        
        real, allocatable, dimension(:,:) :: x, y
        !real, allocatable, dimension(:,:) :: Dd

        real count
        
        integer :: i, j

        count = -1.0

        allocate(x(width,width))
        allocate(y(width,width))
        !allocate(Dd(width,width))

        do j = 1, width !j = coluna
            do i = 1, width !i = linha
                x(i,j) = count !para todas as linhas da coluna j
            enddo
            count = count + (2/(real(width-1)))
        enddo

        y = transpose(x)

        D = real(height)*exp(-5*(x**2+y**2)) !para todas as linha da coluna j

        !D = Dd

        deallocate(x)
        deallocate(y)
        !deallocate(Dd)

    END SUBROUTINE !droplet

    !***********************************************************************
    ! Traducao da funcao modelo2D do Matlab para o Fortran
    !SUBROUTINE model2d(dx,dy,dt,ni,nj,nk,q,u,v,Hmean,rq,ru,rv,f,g)
    !***********************************************************************
    SUBROUTINE model2d(dx, dy, dt, ni, nj, Hmean, rq, ru, rv, f, g, qGl, uGl, vGl)
        IMPLICIT NONE
        double precision :: dx, dy        
        integer, intent(in) :: dt
        integer, intent(in) :: ni, nj
        integer, intent(in) :: Hmean
        double precision, intent(in) :: rq, ru, rv, f, g
        double precision, intent(inout) :: qGl(:,:), uGl(:,:), vGl(:,:)
        
        double precision :: rho_a, rho_w
        double precision :: cx, cy
        double precision :: Cd
        double precision :: Fv, Fu
        double precision :: ua

        double precision, allocatable, dimension(:,:) :: divx, divy
        double precision, allocatable, dimension(:,:) :: dqdx, dqdy, aux
        double precision, allocatable, dimension(:,:) :: ubar, vbar
        
        integer :: i, j

        allocate(divx(ni,nj))
        allocate(divy(ni,nj))
        allocate(dqdx(ni,nj))
        allocate(aux(ni,nj))
        allocate(dqdy(ni,nj))
        allocate(ubar(ni,nj))
        allocate(vbar(ni,nj))

        cx = 1.0/dx
        cy = 1.0/dy

        Cd = 1.6d-3
        rho_a = 1.275
        rho_w = 1.0d3
        ua = 5.0d0

        Fu = Cd * rho_a * ua * ua / (real(Hmean) * rho_w)
        Fv = 0.0

        divx=0.0
        divy=0.0
        dqdx=0.0
        dqdy=0.0

!$OMP PARALLEL            &
!$OMP DEFAULT(shared)     &
!$OMP PRIVATE(i,j)        
        
        !Calculo dos divergentes na direcao x
!$OMP DO
        do j = 1, nj - 1
!DIR$ IVDEP 
           do i = 1, ni - 1 
                divx(i,j)  = cx * (uGl(i+1, j) - uGl(i, j))
            enddo
        enddo
!$OMP END DO

!$OMP DO
        do j = 1, nj - 1
            divx(ni,j) = cx * (uGl(1,j) - uGl(ni,j))
        enddo
!$OMP END DO 

        !Calculo dos divergentes na direcao y
!$OMP DO        
        do j = 1, nj - 1 !j = coluna
!DIR$ IVDEP 
            do i = 1, ni !i = linha
                divy(i, j)  = cy * (vGl(i, j+1) - vGl(i, j))
            enddo
        enddo
!$OMP END DO

!$OMP DO        
        do j = 1, nj - 1 !j = coluna
            divx(ni, j) = cx * (uGl(1, j) - uGl(ni, j))
        enddo
!$OMP END DO

!$OMP DO
        do j = 1, nj-1 !j = coluna
!DIR$ IVDEP 
            do i = 1, ni !i = linha
                qGl(i,j) = qGl(i,j) + real(dt) *(-(real(Hmean)) * (divx(i,j) + divy(i,j)) - (rq * qGl(i,j)))
            enddo
        enddo
!$OMP END DO

!$OMP DO
        do j = 1, nj-1 ! j = coluna
!DIR$ IVDEP 
            do i = 2, ni !i = linha
                dqdx(i,j) = cx * (qGl(i,j) - qGl(i-1,j))
            enddo
        enddo
!$OMP END DO

!$OMP DO        
        do j = 1, nj-1 ! j = coluna
            dqdx(1,j) = cx * (qGl(1,j) - qGl(ni,j))
        enddo
!$OMP END DO

!$OMP DO
        do j = 2, nj-1 !j = coluna
!DIR$ IVDEP 
            do i = 1, ni !i = linha
                dqdy(i,j)= cy * (qGl(i,j) - qGl(i,j-1))
            enddo
        enddo
!$OMP END DO

!$OMP DO
        do j = 2, nj-1!j = coluna
!DIR$ IVDEP 
            do i = 1, ni-1 !i = linha
                ubar(i,j) = 0.25 * (uGl(i+1,j) + uGl(i,j) + uGl(i+1,j-1) + uGl(i,j-1))
            enddo
        enddo
!$OMP END DO

!$OMP DO        
        do j = 2, nj-1!j = coluna
            ubar(ni,j) = 0.25 * (uGl(1,j) + uGl(ni,j) + uGl(1,j-1) + uGl(ni,j-1))
        enddo
!$OMP END DO

!$OMP DO
        do j = 1, nj-1 !i = linha
!DIR$ IVDEP 
            do i = 2, ni !j = coluna
                vbar(i,j) = 0.25 * (vGl(i,j+1) + vGl(i,j) + vGl(i-1,j+1) + vGl(i-1,j))
            enddo
        enddo
!$OMP END DO

!$OMP DO        
        do j = 1, nj-1 !i = linha
            vbar(1,j) = 0.25 * (vGl(1,j+1) + vGl(1,j) + vGl(ni,j+1) + vGl(ni,j))
        enddo
!$OMP END DO

!$OMP DO        
        !Atualizando u
        do j = 1, nj-1 !j = coluna
!DIR$ IVDEP 
            do i  = 1, ni !i = linha
                uGl(i,j) = uGl(i,j) + real(dt) *(f * vbar(i,j) - g * dqdx(i,j) - ru * uGl(i,j) + Fu)
            enddo
        enddo
!$OMP END DO

        !Atualiza v
!$OMP DO        
        do j = 2, nj-1 !j= coluna
!DIR$ IVDEP 
            do i = 1, ni !i = linha    
                vGl(i,j) = vGl(i,j) + real(dt) * (-f * ubar(i,j) - g * dqdy(i,j) - rv * vGl(i,j) + Fv)
            enddo
        enddo
!$OMP END DO


!$OMP END PARALLEL

        deallocate(divx)
        deallocate(divy)
        deallocate(dqdx)
        deallocate(aux)
        deallocate(dqdy)
        deallocate(ubar)
        deallocate(vbar)

    END SUBROUTINE !model2d

    !***********************************************************************
    ! Returns the inverse of a matrix calculated by finding the LU
    ! decomposition.  Depends on LAPACK.
    !***********************************************************************
    function inv(A) result(Ainv)
        use, intrinsic :: iso_fortran_env
        integer, parameter :: sp = REAL32
        integer, parameter :: dp = REAL64
        integer, parameter :: qp = REAL128

        real(dp), dimension(:,:), intent(in) :: A
        real(dp), dimension(size(A,1),size(A,2)) :: Ainv

        real(dp), dimension(size(A,1)) :: work  ! work array focr LAPACK
        integer, dimension(size(A,1)) :: ipiv   ! pivot indices
        integer :: n, info

        ! External procedures defined in LAPACK
        external DGETRF
        external DGETRI

        ! Store A in Ainv to prevent it from being overwritten by LAPACK
        print*, 'Store A in Ainv'
        Ainv = A
        print*,'Ainv = A'
        n = size(A,1)

        ! DGETRF computes an LU factorization of a general M-by-N matrix A
        ! using partial pivoting with row interchanges.
        call DGETRF(n, n, Ainv, n, ipiv, info)
        print*,'call DGETRF'

        if (info /= 0) then
            stop 'Matrix is numerically singular!'
        end if

        ! DGETRI computes the inverse of a matrix using the LU factorization
        ! computed by DGETRF.
        call DGETRI(n, Ainv, n, ipiv, work, n, info)

        if (info /= 0) then
            stop 'Matrix inversion failed!'
        end if
    end function inv

END MODULE
