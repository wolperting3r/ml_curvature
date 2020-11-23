#include "listdefines.h"
!#####################################################################
subroutine usrini(ngr,adapdt, &
#                 include "listall.h"  
                  ier)       
!#####################################################################
! User routine called once on every grid level before the simulation.
!#####################################################################
use adaptime_mod
use ml_mod
implicit none
#include "cb3dall.h"
#include "cbglobaldim.h"

! Import machine learning global variables

! dummy variables
! Variables of the adaptative time step
type(adaptime_typ), intent(in out) :: adapdt
integer, intent(in) :: ngr
#include "dimall.h"
integer :: ier

! local variables
integer :: i,j,k,m,inp,ittemp,itimes
integer :: icphi,incst,incst1
real*8 :: fden,fden2,fvis,fvis2

! local variables for VOF distribution
real*8 :: xr,yr,zr,rb,rc,franzx(8),franzcount,phi(4),phic,FranzA,FranzB1,FranzB2,FranzA2
real*8 :: schenk(6),Farea,ellipse1,ellipse2
real*8 :: r(4),r1,radius1,radius2

integer :: temp1
integer :: stat  ! status of file io
integer                                 :: ml_totalweightlength
integer                                 :: ml_totalbiaslength
integer                                 :: ml_linecounter_layer
integer                                 :: ml_linecounter_bias

! arguments
integer                             :: npts(3)


! local variables
real*8, dimension(:,:), allocatable :: gridpoints
real*8                              :: vertices(8,3)
real*8                              :: radius,center_x,center_y,center_z
real*8                              :: a000,ax00,a0y0,axy0,a00z,ax0z,a0yz,axyz
integer                             :: idiv,jdiv,kdiv,ipts
integer                             :: nidiv,njdiv,nkdiv
integer                             :: tnpts,tndiv,nptsInside
character(len=7)                    :: typeOfObject
real*8                              :: xcdiff,firstdev,seconddev
#include "hahihjhk.h"
!#######################################################################
call setzero(c,nxyzac,1,nxyzac)
call setzero(cold,nxyzac,1,nxyzac)

loop_counter = 1
open(unit=142,file=&
  'model_mlp_750_200-100-25_7x7_rot_norm_10_dshift1_shift1_ed2_s2',&
  action='read',status='old',iostat=stat)
if (stat == 0) then
        allocate(ml_tmp_curv(128, 128))
        allocate(ml_last_curv(128, 128))
        ! initialize ml_tmp_curv
        do m=1,nblo
        call setind(ngr,m, &
#                   include "listsetind.h"
                    ier)
          do k=2,nkm
            do i=2,nim
              do j=2,njm
                ml_tmp_curv(i-1, j-1) = 0
                ml_last_curv(i-1, j-1) = 0
              end do 
            end do 
          end do 
        end do 

        
        read(142,*) stz                 ! stencil size
        read(142,*) ml_n_features       ! number of input features
        read(142,*) ml_n_layers         ! number of layers (including input, including output)
        ! Input layer is a layer too
        ml_n_layers = ml_n_layers+1

        ! Allocate global variables 
        allocate(ml_input(ml_n_features))               ! input features
        allocate(ml_layer_nodes(ml_n_layers))           ! vector with number of nodes
        allocate(ml_layer_activation(ml_n_layers))      ! vector with activation function


        ! First layer is input layer
        ml_layer_nodes(1)=ml_n_features
        ! Read number of nodes per layer
        do i=2,ml_n_layers
          read(142,*) ml_layer_nodes(i)
        end do

        ! First layer has no activation function
        ml_layer_activation(1)='line'
        ! Read activation function per layer
        do i=2,ml_n_layers
          read(142,*) ml_layer_activation(i)
        end do
        
        ! Read bias switch
        read(142,*) ml_bias_on
        ! Read edge switch
        read(142,*) ml_edge_on
        ! Read unsharp masking switch
        read(142,*) ml_mask_on
        ! read(142,*) ml_out_bias_on

        ! Allocate variables needed for edge
        if (ml_edge_on == 1) then
          ! Two times (stz-2)^2 instead of one time stz^2
          ! ml_layer_nodes(1) = 2*(stz-2)**2

          allocate(ml_mask(stz, stz))
          allocate(ml_mask_x((stz-2), (stz-2)))
          allocate(ml_mask_y((stz-2), (stz-2)))
          allocate(ml_and_mask_x((stz-2), (stz-3)))
          allocate(ml_and_mask_y((stz-3), (stz-2)))
          allocate(ml_sum_mask_x((stz-2)))
          allocate(ml_sum_mask_y((stz-2)))
          allocate(ml_sum_and_mask_x((stz-3)))
          allocate(ml_sum_and_mask_y((stz-3)))

          allocate(ml_interp_x((stz-2), (stz-2)))
          allocate(ml_interp_y((stz-2), (stz-2)))
          allocate(ml_data_cut((stz-2), (stz-2)))
          allocate(ml_mask_cut((stz-2), (stz-2)))
          allocate(ml_data_x((stz-2), (stz-2)))
          allocate(ml_data_y((stz-2), (stz-2)))
          allocate(ml_row((stz-2)))
          allocate(ml_and_row((stz-3)))
        else if (ml_mask_on == 1) then
          ! Variables needed for unsharp masking
          ! allocate(ml_stencil(stz, stz))
          ! allocate(ml_interface_mask(stz+2, stz+2))
          ! allocate(ml_nb_mask(stz, stz))
          ! allocate(ml_unsharp_mask(stz, stz))
          ! allocate(ml_difference(stz, stz))
          ! allocate(ml_padded_stencil(stz+2, stz+2))
        end if

        ! Calculate number of layer weights
        allocate(ml_n_layerweights(ml_n_layers))
        if (ml_bias_on == 1) then
          allocate(ml_n_biasweights(ml_n_layers))
        end if

        ! Input layer has no weights
        ml_n_layerweights(1) = 0
        ! Tensors between layers
        do i=2,ml_n_layers
          ml_n_layerweights(i) = ml_layer_nodes(i-1)*ml_layer_nodes(i)
        end do
        ! Calculate total length of node weights
        ml_totalweightlength = 0
        do i=1,ml_n_layers
          ml_totalweightlength = ml_n_layerweights(i) + ml_totalweightlength
        end do
        ! Allocate node weight arrays with total length
        allocate(ml_layerweights(ml_totalweightlength))

        ! Do the same for biases
        if (ml_bias_on == 1) then
          ! Input layer has no biases
          ml_n_biasweights(1) = 0
          ! Biases are added to each hidden layer
          do i=2,ml_n_layers
            ml_n_biasweights(i) = ml_layer_nodes(i)
          end do
          ! Calculate total length of bias weights
          ml_totalbiaslength = 0
          do i=1,ml_n_layers
            ml_totalbiaslength = ml_n_biasweights(i) + ml_totalbiaslength
          end do
          ! Allocate bias weight arrays with total length
          allocate(ml_biasweights(ml_totalbiaslength))
        end if

        ml_linecounter_layer = 0
        ml_linecounter_bias = 0
        do i=1,ml_n_layers  ! Iterate over every layer
          ! Read node weights of layer i
          do j=1,ml_n_layerweights(i)  ! Iterate over all node weights of layer i
            read(142,*) ml_layerweights(ml_linecounter_layer+j) 
          end do
          ml_linecounter_layer = ml_linecounter_layer + ml_n_layerweights(i)

          ! Read bias weights of layer i
          if (ml_bias_on == 1) then
            do j=1,ml_n_biasweights(i)
              read(142,*) ml_biasweights(ml_linecounter_bias+j) 
            end do
            ml_linecounter_bias = ml_linecounter_bias + ml_n_biasweights(i)
          end if

        end do

        ! Allocate layers
        ! allocate(ml_input_layer(ml_layer_nodes(1)))
        ! allocate(ml_first_layer(ml_layer_nodes(2)))
        ! allocate(ml_second_layer(ml_layer_nodes(3)))
        ! allocate(ml_third_layer(ml_layer_nodes(4)))
        ! allocate(ml_output_layer(ml_layer_nodes(5)))
        ! allocate(ml_tmp(ml_layer_nodes(1)))
        print*, 'file ml_weights was loaded'
else
        print*, 'file ml_weights not found'
end if


npts(1:3)=10
icphi=1
incst=icphi*nxyza

!circle or sphere
typeOfObject = 'ellipse'

! Process input variables
tnpts = npts(1)*npts(2)*npts(3)  ! total number of points
nidiv = npts(1)-1                ! number of divisions in i
njdiv = npts(2)-1                ! number of divisions in j
nkdiv = npts(3)-1                ! number of divisions in k
tndiv = nidiv*njdiv*nkdiv        ! total number of divisions

! Allocate and initialize arrays
allocate(gridpoints(tnpts,3))

! Circle/Sphere parameter
radius1  = 0.016893d0!0.012d0!0.021d0!0.016893d0!0.01762d0
radius2  = 0.028226d0!0.030d0!0.02d0!0.0275d0!
center_x = 0.0375d0
center_y = 0.0375d0
center_z = 0.0

do m=1,nblo
  call setind(1, m, &
#             include "listsetind.h"
              ier)
  do k=2,nkm
    do i=2,nim
      do j=2,njm
        inp=ha(i,j,k)
        ! Get vertices of the current control volume
        vertices(1,1) = x(ha(i-1,j-1,k-1))
        vertices(1,2) = y(ha(i-1,j-1,k-1))
        vertices(1,3) = z(ha(i-1,j-1,k-1))
        vertices(2,1) = x(ha(i  ,j-1,k-1))
        vertices(2,2) = y(ha(i  ,j-1,k-1))
        vertices(2,3) = z(ha(i  ,j-1,k-1))
        vertices(3,1) = x(ha(i-1,j  ,k-1))
        vertices(3,2) = y(ha(i-1,j  ,k-1))
        vertices(3,3) = z(ha(i-1,j  ,k-1))
        vertices(4,1) = x(ha(i  ,j  ,k-1))
        vertices(4,2) = y(ha(i  ,j  ,k-1))
        vertices(4,3) = z(ha(i  ,j  ,k-1))
        vertices(5,1) = x(ha(i-1,j-1,k  ))
        vertices(5,2) = y(ha(i-1,j-1,k  ))
        vertices(5,3) = z(ha(i-1,j-1,k  ))
        vertices(6,1) = x(ha(i  ,j-1,k  ))
        vertices(6,2) = y(ha(i  ,j-1,k  ))
        vertices(6,3) = z(ha(i  ,j-1,k  ))
        vertices(7,1) = x(ha(i-1,j  ,k  ))
        vertices(7,2) = y(ha(i-1,j  ,k  ))
        vertices(7,3) = z(ha(i-1,j  ,k  ))
        vertices(8,1) = x(ha(i  ,j  ,k  ))
        vertices(8,2) = y(ha(i  ,j  ,k  ))
        vertices(8,3) = z(ha(i  ,j  ,k  ))

        ! Create points distributed evenly in the current control volume
        ipts = 1
        do kdiv=0,nkdiv
          do jdiv=0,njdiv
            do idiv=0,nidiv
              a000 = real((nidiv-idiv) * (njdiv-jdiv) * (nkdiv-kdiv),8) / real(tndiv,8)
              ax00 = real((      idiv) * (njdiv-jdiv) * (nkdiv-kdiv),8) / real(tndiv,8)
              a0y0 = real((nidiv-idiv) * (      jdiv) * (nkdiv-kdiv),8) / real(tndiv,8)
              axy0 = real((      idiv) * (      jdiv) * (nkdiv-kdiv),8) / real(tndiv,8)
              a00z = real((nidiv-idiv) * (njdiv-jdiv) * (      kdiv),8) / real(tndiv,8)
              ax0z = real((      idiv) * (njdiv-jdiv) * (      kdiv),8) / real(tndiv,8)
              a0yz = real((nidiv-idiv) * (      jdiv) * (      kdiv),8) / real(tndiv,8)
              axyz = real((      idiv) * (      jdiv) * (      kdiv),8) / real(tndiv,8)

              gridpoints(ipts,1) = a000*vertices(1,1) + ax00*vertices(2,1) &
                                 + a0y0*vertices(3,1) + axy0*vertices(4,1) &
                                 + a00z*vertices(5,1) + ax0z*vertices(6,1) &
                                 + a0yz*vertices(7,1) + axyz*vertices(8,1)
              gridpoints(ipts,2) = a000*vertices(1,2) + ax00*vertices(2,2) &
                                 + a0y0*vertices(3,2) + axy0*vertices(4,2) &
                                 + a00z*vertices(5,2) + ax0z*vertices(6,2) &
                                 + a0yz*vertices(7,2) + axyz*vertices(8,2)
              gridpoints(ipts,3) = a000*vertices(1,3) + ax00*vertices(2,3) &
                                 + a0y0*vertices(3,3) + axy0*vertices(4,3) &
                                 + a00z*vertices(5,3) + ax0z*vertices(6,3) &
                                 + a0yz*vertices(7,3) + axyz*vertices(8,3)

              ipts = ipts + 1
            end do
          end do
        end do

        ! Check how many points are inside the circle/sphere
        nptsInside = 0
        select case(typeOfObject)
          case('circle')
            do ipts=1,tnpts
              if (((gridpoints(ipts,1)-center_x)**2+(gridpoints(ipts,2)-center_y)**2) < radius**2) then
                nptsInside = nptsInside+1
              end if
            end do
          case('ellipse')
            do ipts=1,tnpts
              if ((((gridpoints(ipts,1)-center_x)/radius1)**2+((gridpoints(ipts,2)-center_y)/radius2)**2 ) <= 1.d0) then
                nptsInside = nptsInside+1
              end if
            end do
          case('sphere')
            do ipts=1,tnpts
              if (((gridpoints(ipts,1)-center_x)**2 &
                  +(gridpoints(ipts,2)-center_y)**2 &
                  +(gridpoints(ipts,3)-center_z)**2) < radius**2) then
                nptsInside = nptsInside+1
              end if
            end do
        end select

        ! Calculate volume fraction
        c(inp) = (real(nptsInside,8) / real(tnpts,8))
        cold(inp)=c(inp)
        c(inp+incst)=1d0-c(inp)
        cold(inp+incst)=1d0-c(inp)

        !p(inp)=dskappa/radius

        !cm(inp)=c(inp)
        !cm(inp+incst)=c(inp+incst)

      end do
    end do
  end do
end do

deallocate(gridpoints)

! do m=1,nblo
!     call setind(1,m, &
! #               include "listsetind.h"            
!                 ier)            
!   do k=2,nkm
!     do i=2,nim
!       do j=2,njm
!         inp=ha(i,j,k)
!          if (c(inp) > 0.01d0 .or. c(inp) < 0.99d0) then
!            xcdiff=xc(inp)-center_x
!            firstdev =-radius1/radius2 *(xcdiff) *1/sqrt(radius2-xcdiff**2)
!            seconddev=-radius1/radius2 *( xcdiff**2/(radius2**2-xcdiff**2)**(3/2)  - 1/(radius2**2-xcdiff**2)**(1/2) )
!            kappaana(inp)=seconddev(inp)/ (1+firstdev**2)**(3/2)
!          end if
!        end do
!      end do
!    end do
!  end do

return
end subroutine usrini

!#####################################################################
subroutine usrgri( &
#                 include "listall.h" 
                  ier)          
!#####################################################################
! User routine called before changing to a different grid (level).
!#####################################################################
implicit none
#include "cb3dall.h"
#include "cbglobaldim.h"

! dummy variables
#include "dimall.h"
integer :: ier

! local variables
#include "hahihjhk.h"
!#####################################################################

return
end


!#####################################################################
subroutine usrtim( &
#                 include "listall.h" 
                  ier)          
!#####################################################################
! User routine called before each time step.
!#####################################################################
implicit none
#include "cb3dall.h"
#include "cbglobaldim.h"

! dummy variables
#include "dimall.h"
integer :: ier

! local variables
#include "hahihjhk.h"
!#####################################################################

return
end


!#####################################################################
subroutine util(adapdt, &
#               include "listall.h"
                ier)           
!#####################################################################
! User routine called after each time step. For steady calculations called at the end of each grid level (FMG).
!#####################################################################
use adaptime_mod
implicit none
#include "cb3dall.h"
#include "cbglobaldim.h"

! Variables of the adaptative time step
type(adaptime_typ), intent(in out) :: adapdt
! dummy variables
#include "dimall.h"
integer :: ier

integer :: inp,i,j,k,m
real*8  :: xmin,xmax,ymin,ymax,zmin,zmax
real*8  :: xmintemp,xmaxtemp,ymintemp,ymaxtemp,zmintemp,zmaxtemp
character(len=20) :: filename,filename2
real,dimension(3) :: l_start,l_end,l_inc,l_point
real,dimension(4) :: mindis,mindisvec
integer,dimension(4) :: mininp
real :: currdis,l_probe,mindisnom
integer :: n_points,l_i,min_n

real*8 :: xc1,xc2,yc1

#include "hahihjhk.h"
!#####################################################################
if (it == 1 .and. iproc==1) then
open(69,file="summary.txt",position='append',status='replace')
write(69,456) 'it','innerit','res u','res v','res w','res p','res conc','time timestep','time all'
close(69)
456   FORMAT(12(A, 8X))
end if      

if(iproc==1) then
open(69,file="summary.txt",position='append')
write(69,457) it,ls,resor(iu),resor(iv),resor(iw),resor(ip),resor(icon),deltim,tetime
close(69)
457   FORMAT(i4,2x,i4,2x,e11.4,2x,e11.4,2x,e11.4,2x,e11.4,2x,e11.4,2x,f15.5,2x,f15.5)
end if


! For specific output in the moving bubble testcase
if (it == 1 .and. iproc==1) then
open(69,file="y_pos.txt",position='append',status='replace')
write(69,458) 'it','x','y','c','cm'
close(69)
458   FORMAT(5(A, 5X))
end if      


do m=1,nblo
    call setind(1,m, &
#             include "listsetind.h"
              ier)

do k=2,nkm
  do i=2,nim
    do j=2,njm
      inp=ha(i,j,k) 
      yc1=0.0375d0
      if (nj<100) then
        xc1=0.037d0 
        xc2=0.036d0
        !!!!  NICHT SCHÖN SO !!!
      else if (nj<200 .and. nj>100 .and. nproc==1) then
        xc1=0.0378d0
        xc2=0.0373d0
      else if (nj<300 .and. nj>200 .or. nproc>1) then
        xc1=0.03751d0
        xc2=0.03722d0
      end if
      if (xc(inp)<xc1 .and. xc(inp)>xc2 .and. yc(inp)>yc1 .and.k==2) then      
        open(69,file="y_pos.txt",position='append')
        write(69,*) it, xc(inp),yc(inp),c(inp),cm(inp)
        close(69)
      endif
     end do
   end do
end do
end do

! ! Compare normals
! center_x=0.0375d0
! center_y=0.0375d0
! lmax_x=0d0
! lmax_y=0d0
! do m=1,nblo
!     call setind(1,m, &
! #               include "listsetind.h"            
!                 ier)            
!   do k=2,nkm
!     do i=2,nim
!       do j=2,njm
!         inp=ha(i,j,k)
!         !Vector from center to points
!         vecx=(center_x-xc(inp))
!         vecy=(center_y-yc(inp))
!         normalized=sqrt(vecx**2+vecy**2)
!         realnormalx(inp)=vecx/normalized
!         realnormaly(inp)=vecy/normalized
!         normalized=sqrt(dscdx(inp)**2+dscdy(inp)**2)
!         newnormalx(inp)=dscdx(inp)/normalized
!         newnormaly(inp)=dscdy(inp)/normalized
!         lmax_xtemp(inp)=abs(realnormalx(inp)-newnormalx(inp))
!         lmax_ytemp(inp)=abs(realnormaly(inp)-newnormaly(inp))
!         if (c(inp)<0.99d0 .and. c(inp)>0.01d0) then
!           if (lmax_xtemp(inp)>lmax_x) then
!             lmax_x=lmax_xtemp(inp)
!           endif
!           if (lmax_ytemp(inp)>lmax_y) then
!             lmax_y=lmax_ytemp(inp)
!           endif
!         endif
!         !if (c(inp)<0.99d0 .and. c(inp)>0.01d0) then
!            !print *,c(inp),cm(inp),vecx,dscdx(inp),vecy,dscdy(inp)
!          !  print *,c(inp),cm(inp),realnormalx(inp),newnormalx(inp),realnormaly(inp),newnormaly(inp)
!         !end if
!         !if (lmax_x > 1.9999d0) then
!         !  STOP
!         !end if
!       end do
!     end do
!   end do
! end do
  
return
end

!#####################################################################
subroutine usritr( &
#                 include "listall.h" 
                  ier)          
!#####################################################################
! User routine called before every V-cycle.
!#####################################################################
implicit none
#include "cb3dall.h"
#include "cbglobaldim.h"

! dummy variables
#include "dimall.h"
integer :: ier

! local variables
#include "hahihjhk.h"
!#####################################################################

return
end


!#####################################################################
subroutine usrlsg( &
#                 include "listall.h"  
                  ier)         
!#####################################################################
! User routine called before every fine grid iteration of a V-cycle.
!#####################################################################
implicit none
#include "cb3dall.h"
#include "cbglobaldim.h"

! dummy variables
#include "dimall.h"
integer :: ier

! local variables
#include "hahihjhk.h"
!#####################################################################

return
end
