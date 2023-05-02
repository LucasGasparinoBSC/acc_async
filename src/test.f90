program test
    use cudafor
    use openacc
    use mod_nvtx

    implicit none
    integer(8)              :: ielem, inode, ipoin, igaus, iter
    integer(8)              :: nelem, nnode, npoin, ngaus
    integer(8), allocatable :: meshTable(:,:)
    real(4)                 :: aux_c, aux_d
    real(4)   , allocatable :: Rconvec(:), Rdiffu(:), u_g(:), u_l(:), Re_c(:), Re_d(:), shape_n(:,:)

    ! Initialize pseudo-mesh props
    nelem = 1000000
    nnode = 64
    ngaus = 64
    npoin = nnode*nelem
    allocate(meshTable(nelem,nnode))

    ! Create meshTable
    call nvtxStartRange("meshTable")
    !$acc enter data create(meshTable)
    !$acc parallel loop gang present(meshTable)
    do ielem = 1,nelem
        !$acc loop vector
        do inode = 1,nnode
            meshTable(ielem,inode) = (ielem-1)*nnode + inode
        end do
    end do
    !$acc end parallel loop
    call nvtxEndRange()

    ! Create pseudo shape functions
    allocate(shape_n(ngaus,nnode))

    call nvtxStartRange("shape_n")
    !$acc enter data create(shape_n)
    !$acc kernels present(shape_n)
    shape_n(:,:) = 1.0
    !$acc end kernels
    call nvtxEndRange()

    ! Create and fill initial global data array
    allocate(u_g(npoin))

    call nvtxStartRange("u_g")
    !$acc enter data create(u_g)
    !$acc parallel loop gang vector present(u_g)
    do ipoin = 1,npoin
        u_g(ipoin) = real(ipoin,4)/real(npoin,4)
    end do
    !$acc end parallel loop
    call nvtxEndRange()

    ! Create an initialize global arrays for convection and diffusion
    allocate(Rconvec(npoin))
    allocate(Rdiffu(npoin))

    call nvtxStartRange("Rconvec, Rdiffu")
    !$acc enter data create(Rconvec,Rdiffu)
    !$acc parallel loop gang vector present(Rconvec,Rdiffu)
    do ipoin = 1,npoin
        Rconvec(ipoin) = 0.0
        Rdiffu(ipoin) = 0.0
    end do
    !$acc end parallel loop
    call nvtxEndRange()

    ! Allocate local arrays
    allocate(u_l(nnode))
    allocate(Re_c(nnode))
    allocate(Re_d(nnode))
    !$acc enter data create(u_l,Re_c,Re_d)

    ! Start pseudo-kernels
    call nvtxStartRange("pseudo-kernels")
    do iter = 1,10

        ! Pseudo-convective
        !$acc parallel loop gang present(meshTable,u_g) private(u_l,Re_c) async(1)
        do ielem = 1,nelem
            !$acc loop vector
            do inode = 1,nnode
                u_l(inode) = u_g(meshTable(ielem,inode))
                Re_c(inode) = 0.0
            end do
            !$acc loop seq
            do igaus = 1,ngaus
                aux_c = 0.0
                !$acc loop vector reduction(+:aux_c)
                do inode = 1,nnode
                    aux_c = aux_c + shape_n(igaus,inode)*u_l(inode)
                end do
                !$acc loop vector
                do inode = 1,nnode
                    Re_c(inode) = Re_c(inode) + shape_n(igaus,inode)*aux_c
                end do
            end do
            !$acc loop vector
            do inode = 1,nnode
                !$acc atomic update
                Rconvec(meshTable(ielem,inode)) = Rconvec(meshTable(ielem,inode)) + Re_c(inode)
                !$acc end atomic
            end do
        end do
        !$acc end parallel loop

        ! Pseudo-diffusive
        !$acc parallel loop gang present(meshTable,u_g) private(u_l,Re_d) async(2)
        do ielem = 1,nelem
            !$acc loop vector
            do inode = 1,nnode
                u_l(inode) = u_g(meshTable(ielem,inode))
                Re_d(inode) = 0.0
            end do
            !$acc loop seq
            do igaus = 1,ngaus
                aux_d = 0.0
                !$acc loop vector reduction(+:aux_d)
                do inode = 1,nnode
                    aux_d = aux_d + shape_n(igaus,inode)*u_l(inode)/2.0
                end do
                !$acc loop vector
                do inode = 1,nnode
                    Re_d(inode) = Re_d(inode) + shape_n(igaus,inode)*aux_d/2.0
                end do
            end do
            !$acc loop vector
            do inode = 1,nnode
                !$acc atomic update
                Rdiffu(meshTable(ielem,inode)) = Rdiffu(meshTable(ielem,inode)) + Re_d(inode)
                !$acc end atomic
            end do
        end do
        !$acc end parallel loop

        ! Wait on all streams
        !$acc wait

    ! End pseudo-kernels
    end do
    call nvtxEndRange()

end program test