module ml_subroutines
contains
subroutine ml_edge(ml_stencil, axis, ml_interpolated_dir)
use ml_mod
implicit none
integer, intent(in) :: axis ! 1 = y, 2 = x, 3 = x and y 
! real*8,allocatable,dimension(:, :),intent(in, out) :: ml_interpolated_dir

integer :: counter1, counter2, search_counter
real*8,allocatable,dimension(:, :) :: ml_stencil, ml_pairs, ml_interpolated, ml_intp_searchresult, ml_intp_mask
real*8,allocatable,dimension(:)    :: ml_intp_searchfor, ml_last_layer_output, ml_interpolated_dir
integer :: i_loc, j_loc

! For test only
! real*8,allocatable,dimension(:)    :: ml_test_tmp

if (1 == 1) then ! Edge2 switch
  ! Initialize ml_pairs array (j_p1, i_p1, j_p2, i_p2, concentration_p1, concentration_p2, interp)
  allocate(ml_pairs(stz**2, 7))
  do j_loc=1,7
    do i_loc = 1, stz**2
      ml_pairs(i_loc, j_loc) = 0
    end do
  end do
  counter1 = 0
  ! Find indices of values where concentration > 0.5 and the neighbours concentration is < 0.5 (pair). The indices refer to the higher value, the index of the lower value depends on the position (north, south, east, west).
  ! Iterate over rows
  do j_loc=1,stz
    ! Iterate over columns
    do i_loc=1,stz
      ! Only continue if concentration >= 0.5 (potential upper partner of pair containing c = 0.5)
      if (ml_stencil(j_loc, i_loc) >= 0.5) then
        if ((axis == 1) .or. (axis == 3)) then ! North-South (y) 
          ! Only look for northern partner if j_loc, i_loc is not in top row
          if (j_loc < stz) then
            ! Check if northern cell satisfies concentration < 0.5 (lower partner)
            if (ml_stencil(j_loc+1, i_loc) < 0.5) then ! North
              ! i, j and northern neighbour are a pair -> save indices and values in ml_pairs
              counter1 = counter1 + 1
              ml_pairs(counter1, 1) = j_loc ! Indices upper partner
              ml_pairs(counter1, 2) = i_loc
              ml_pairs(counter1, 3) = j_loc+1 ! Indices lower partner
              ml_pairs(counter1, 4) = i_loc
              ml_pairs(counter1, 5) = ml_stencil(j_loc, i_loc) ! Value upper partner
              ml_pairs(counter1, 6) = ml_stencil(j_loc+1, i_loc) ! Value lower partner
            end if
          end if
          ! Only look for southern partner if j_loc, i_loc is not in bottom row
          if (j_loc > 1) then
            if (ml_stencil(j_loc-1, i_loc) < 0.5) then ! South
              ! i, j and southern neighbour are a pair
              counter1 = counter1 + 1
              ml_pairs(counter1, 1) = j_loc ! Indices upper partner
              ml_pairs(counter1, 2) = i_loc
              ml_pairs(counter1, 3) = j_loc-1 ! Indices lower partner
              ml_pairs(counter1, 4) = i_loc
              ml_pairs(counter1, 5) = ml_stencil(j_loc, i_loc) ! Value upper partner
              ml_pairs(counter1, 6) = ml_stencil(j_loc-1, i_loc) ! Value lower partner
            end if
          end if
        end if
        if ((axis == 2) .or. (axis == 3)) then ! East-West (x)
          if (i_loc < stz) then
            if (ml_stencil(j_loc, i_loc+1) < 0.5) then ! East
              ! i, j and eastern neighbour are a pair
              counter1 = counter1 + 1
              ml_pairs(counter1, 1) = j_loc ! Indices upper partner
              ml_pairs(counter1, 2) = i_loc
              ml_pairs(counter1, 3) = j_loc ! Indices lower partner
              ml_pairs(counter1, 4) = i_loc+1 
              ml_pairs(counter1, 5) = ml_stencil(j_loc, i_loc) ! Value upper partner
              ml_pairs(counter1, 6) = ml_stencil(j_loc, i_loc+1) ! Value lower partner
            end if
          end if
          if (i_loc > 1) then
            if (ml_stencil(j_loc, i_loc-1) < 0.5) then ! West
              ! i, j and western neighbour are a pair
              counter1 = counter1 + 1
              ml_pairs(counter1, 1) = j_loc ! Indices upper partner
              ml_pairs(counter1, 2) = i_loc
              ml_pairs(counter1, 3) = j_loc ! Indices lower partner
              ml_pairs(counter1, 4) = i_loc-1
              ml_pairs(counter1, 5) = ml_stencil(j_loc, i_loc) ! Value upper partner
              ml_pairs(counter1, 6) = ml_stencil(j_loc, i_loc-1) ! Value lower partner
            end if
          end if
        end if
      end if
    end do
  end do
  ! Initialize ml_interpolated (j, i, interpolated concentration)
  allocate(ml_interpolated(stz**2, 3))
  do i_loc=1,3
    do j_loc=1,stz**2
      ml_interpolated(j_loc, i_loc) = 0
    end do
  end do
  ! Iterate over ml_pairs to get interpolated concentration of all pairs
  do i_loc=1,counter1
    ! Interpolate
    ml_pairs(i_loc, 7) = (0.5d0 - ml_pairs(i_loc, 6))/(ml_pairs(i_loc, 5) - ml_pairs(i_loc, 6))
    if (ml_pairs(i_loc, 7) >= 0.5d0) then
      ! if interpolated value is >= 0.5, the value of the cell with higher concentration should be 1.5 - interpolated value
      ml_interpolated(i_loc, 1) = ml_pairs(i_loc, 1) ! Indices upper partner
      ml_interpolated(i_loc, 2) = ml_pairs(i_loc, 2)
      ml_interpolated(i_loc, 3) = 1.5 - ml_pairs(i_loc, 7) ! Interpolated value for upper partner
    else
      ! if interpolated value is < 0.5, the value of the cell with lower concentration should be 0.5 - interpolated value
      ml_interpolated(i_loc, 1) = ml_pairs(i_loc, 3) ! Indices lower partner
      ml_interpolated(i_loc, 2) = ml_pairs(i_loc, 4)
      ml_interpolated(i_loc, 3) = 0.5 - ml_pairs(i_loc, 7) ! Interpolated value for lower partner
    end if
  end do

  allocate(ml_intp_searchfor(3))
  ! Initialize array with results that contain every index (j, i) only once and the corresponding concentration (either directly copied from ml_interpolated if index occures just once or by taking the mean of the concentration of all occurances 
  allocate(ml_intp_searchresult(stz**2, 3))
  do i_loc=1,3
    ml_intp_searchfor(i_loc) = 0
    do j_loc=1,stz**2
      ml_intp_searchresult(j_loc, i_loc) = 0
    end do
  end do
  ! Reset counter2 for ml_intp_searchresult
  counter2 = 1

  ! Find indices that occur multiple times and resolve those conflicts by taking the mean
  do i_loc=1,counter1
    ml_intp_searchfor = (/ml_interpolated(i_loc, 1), ml_interpolated(i_loc, 2), ml_interpolated(i_loc, 3)/)
    ! Reset search_counter
    search_counter = 0
    searchloop: do j_loc=1,counter1
      ! If indices are the same, continue to take mean
      if (&
        (ml_interpolated(j_loc, 1) == ml_intp_searchfor(1)) .and. &
        (ml_interpolated(j_loc, 2) == ml_intp_searchfor(2)) &
        ) then
        ! If position of indices found is before position of ml_intp_tmp, stop (tuple was already taken care of)
        if (j_loc < i_loc) then
          search_counter = -1
          exit searchloop
        end if
        ! If position of indices found in ml_interpolated is >= position in ml_intp_tmp, remember indices and get sum of concentration
        if (j_loc >= i_loc) then
          ml_intp_searchresult(counter2, 1) = ml_intp_searchfor(1)
          ml_intp_searchresult(counter2, 2) = ml_intp_searchfor(2)
          ml_intp_searchresult(counter2, 3) = ml_intp_searchresult(counter2, 3) + ml_interpolated(j_loc, 3)
          search_counter = search_counter + 1
        end if
      end if
    end do searchloop
    if (search_counter > 0) then
      ! Divide sum of tuples with same indices by count of those to get mean value
      ml_intp_searchresult(counter2, 3) = ml_intp_searchresult(counter2, 3)/search_counter
      counter2 = counter2 + 1
    end if
  end do
  ! Compensate for last addition to counter2
  counter2 = counter2 - 1

  ! ml_intp_searchresult now contains indices and concentration values of all interpolated values that should be written into interpolated stencil

  allocate(ml_intp_mask(stz, stz))
  ! Initialize output stencil by creating a mask that is 1 where c >= 0.5 and 0 elsewhere
  do j_loc=1,stz
    do i_loc=1,stz
      if (ml_stencil(j_loc, i_loc) >= 0.5d0) then
        ml_intp_mask(j_loc, i_loc) = 1d0
      else 
        ml_intp_mask(j_loc, i_loc) = 0d0
      end if
    end do
  end do

  ! Add interpolated values to stencil
  do i_loc=1,counter2
    ml_intp_mask(int(ml_intp_searchresult(i_loc, 1)), int(ml_intp_searchresult(i_loc, 2))) = ml_intp_searchresult(i_loc, 3)
  end do

  ! Substract 1 where concentration values in stencils are > 1
  do j_loc=1,stz
    do i_loc=1,stz
      if (ml_intp_mask(j_loc, i_loc) > 1d0) then
        ml_intp_mask(j_loc, i_loc) = ml_intp_mask(j_loc, i_loc) - 1d0
      end if
    end do
  end do

  ! Print Test stencil
  ! if (axis == 1) then
    ! print*, "y"
  ! else
    ! print*, "x"
  ! end if
  ! do j_loc=1, stz
    ! print"(f6.3, f6.3, f6.3, f6.3, f6.3, f6.3, f6.3)", &
      ! ml_intp_mask(j_loc, 1), & 
      ! ml_intp_mask(j_loc, 2), & 
      ! ml_intp_mask(j_loc, 3), & 
      ! ml_intp_mask(j_loc, 4), & 
      ! ml_intp_mask(j_loc, 5), & 
      ! ml_intp_mask(j_loc, 6), & 
      ! ml_intp_mask(j_loc, 7)
  ! end do
  ! print*, " " 

  deallocate(ml_pairs)
  deallocate(ml_interpolated)
  deallocate(ml_intp_searchfor)
  deallocate(ml_intp_searchresult)

  ! write values into output array
  counter2 = 1
  do j_loc=1,stz
    do i_loc=1,stz
      ml_interpolated_dir(counter2) = ml_intp_mask(j_loc, i_loc)
      counter2 = counter2 + 1
    end do
  end do
  deallocate(ml_intp_mask)
end if
! deallocate(ml_test_tmp)

end subroutine ml_edge
end module ml_subroutines


#include "listdefines.h"
!##################################################################################################
subroutine curv_ml(ngr, &
#                  include "listcalcurv.h"
                   ier)
!##################################################################################################
! Height-Functions-Method
!##################################################################################################
use ml_mod
use ml_subroutines
implicit none
#include "cb3dall.h"
#include "cbglobaldim.h"

! dummy variables
integer, intent(in out) :: ngr
#include "dimbcoef.h"
#include "dimgeom12.h"
#include "dimdscurv.h"
#include "dimdsst.h"
#include "dimconc.h"
#include "dimlilk.h"
#include "dimconcold.h"
#include "dimdivers.h"
#include "dimindex2.h"
#include "dimindex4.h"
#include "dimliglkg.h"
#include "dimlogic4.h"
#include "dimdiscf.h"
#include "dimtayint.h"
#include "dimrhelp3.h"
#include "dimbndcon.h"
#include "dimblopar.h"
#include "dimacoef.h"
#include "dimdsls.h"
#include "dimdslog.h"
#include "tradef.h"
#include "dimiters.h"
#include "dimcderivatives.h"

integer :: ier
      
! local variables
integer :: i,j,k,m,inp !,incst
real*8,allocatable,dimension(:) :: ml_last_layer_output
real*8,allocatable,dimension(:) :: ml_layer_input
real*8,allocatable,dimension(:) :: ml_tmp
real*8,allocatable,dimension(:) :: ml_interpolated_dir
real*8,allocatable,dimension(:, :) :: ml_stencil
integer,allocatable,dimension(:) :: ml_and_and_mask, ml_row_mask_cut
real*8 :: ml_sum, ml_weight, ml_biasweight, rotation, ml_flip_sum_first, ml_flip_sum_last
integer :: ml_loc
real*8 :: ml_sum_rc, ml_mean, ml_mean2
real*8 :: xm, ym
real*8 :: pi = 4 * atan (1.0_8)  ! Define pi
integer, dimension(1, 1) :: ml_case
integer:: stzh
integer:: checksum
integer:: gridsize = 128
integer :: counter, w_idx_beg, b_idx_beg, i_loc, j_loc, n, o, p, counter1, i_shift, j_shift
integer :: export = 0

real*4 :: t1, t2, t3, t4, delta, delta_ml, delta_pre, delta_tmp !FÃr timing
integer :: timing = 0

integer :: icphi
! integer,intent(in)  :: ngr
! real*8, intent (out):: dscurv2a
      
logical :: lfy
! local variables
logical :: laux,lfend,curvfiltering,curvweightening,secondweighting,flatsurfacectrl,nolocalcurv
! integer :: i,j,k,l,m,count1,inp,checkit,zerocount
!real*8  :: vz,xm,ym,zm,h,gh,dscurvsum2
integer :: curvfilter,weightsexp,weightsexp2
real*8  :: tempcurv(nxyza),tempcurv2(nxyza),weights(nxyza),normalized
real*8  :: fi1(nxyza),fi2(nxyza),fi3(nxyza),fi4(nxyza)
integer :: k1,kn,i1,inn,j1,jn,kstart,kend,numa,numb,incst,incst1
! real*8  :: xclim1,xclim2,xclim1_3d,xclim2_3d,scalingfactor
! real*4  :: t1, delta
#include "hahihjhk.h"
!####################################################################################################################################
! Calculate gradient of concentration field for pre-processing
call calgradc(ngr,0,1,c,dscdx,dscdy,dscdz, &
#               include "listcalgradcc.h"
        ier)  

stzh = (stz-1)/2
if (timing == 1) then
  t1 = secnds(0.0)
  delta_ml = 0.0
  delta_pre = 0.0
  delta_tmp = 0.0
end if
do m=1,nblo
  call setind(ngr,m, &
#             include "listsetind.h"
             ier)
  do k=2,nkm
    do j=2,njm
      do i=2,nim
        if (timing == 1) then
          t2 = secnds(0.0)
        end if
        inp=ha(i,j,k)

        xm = sqrt((x(ha(i, j, k)) - x(ha(i-1, j, k)))**2)
        ym = sqrt((y(ha(i, j, k)) - y(ha(i, j-1, k)))**2)

        if (((cm(ha(i  ,j  ,k)) > 0.05d0) .AND. (cm(ha(i  ,j  ,k)) < 0.95d0))) then
          ! MACHINE LEARNING
          ! Input Layer
          allocate(ml_last_layer_output(stz**2))
          do counter=1,stz**2
            ml_last_layer_output(counter) = 0
          end do
          ! Fill input layer (ml_last_layer_output) with vof concentration values
 
          counter = 1
          do j_loc=-stzh,stzh
            do i_loc=-stzh,stzh
              ! Set blocks where c < 0.005 to 0

              if (cm(ha(i+i_loc, j+j_loc, 1)) < 0.005) then
                ml_last_layer_output(counter) = 0
              else if (cm(ha(i+i_loc, j+j_loc, 1)) > 0.995) then
                ml_last_layer_output(counter) = 1
              else
                ml_last_layer_output(counter) = cm(ha(i+i_loc, j+j_loc, 1))
              end if
              counter = counter+1
            end do
          end do

          ! Test stencil
          ! ml_last_layer_output = (/&
          ! 1.000000,  1.000000,  1.000000,  1.000000,  1.000000,  1.000000,  1.000000, &
          ! 0.980276,  0.972590,  0.980461,  0.998333,  1.000000,  1.000000,  1.000000, &
          ! 0.746525,  0.686491,  0.748200,  0.882598,  0.953884,  0.998055,  1.000000, &
          ! 0.081119,  0.067414,  0.083156,  0.195056,  0.525324,  0.869204,  0.990369, &
          ! 0.000000,  0.000000,  0.000000,  0.007871,  0.071581,  0.318827,  0.798286, &
          ! 0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.038244,  0.287939, &
          ! 0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.000000,  0.045838 &
          ! /)
          ! print*, "Input 1:stz"
          ! do j_loc=1, stz
            ! print"(f6.3, f6.3, f6.3, f6.3, f6.3, f6.3, f6.3)", &
              ! ml_last_layer_output((j_loc-1)*stz+1), & 
              ! ml_last_layer_output((j_loc-1)*stz+2), & 
              ! ml_last_layer_output((j_loc-1)*stz+3), & 
              ! ml_last_layer_output((j_loc-1)*stz+4), & 
              ! ml_last_layer_output((j_loc-1)*stz+5), & 
              ! ml_last_layer_output((j_loc-1)*stz+6), & 
              ! ml_last_layer_output((j_loc-1)*stz+7)
          ! end do

          if (1 == 1) then  ! rotation switch
            allocate(ml_tmp(stz**2))
            do counter=1,stz**2
              ml_tmp = 0
            end do
            ! Calculate quadrant of normal vector 
            ! rotation = floor(((atan(-dscdy(inp), -dscdx(inp))+pi)*1d0/(2d0*pi))*8d0)
            rotation = floor( &
              (atan( &
                -((ml_last_layer_output(stz*(stz-3)/2+(stz-1)/2+1) - ml_last_layer_output(stz*(stz+1)/2+(stz-1)/2+1))/2), &
                -((ml_last_layer_output(stz*(stz-1)/2+(stz-1)/2+2) - ml_last_layer_output(stz*(stz-1)/2+(stz-1)/2))/2) &
              )+pi)*8d0/(2d0*pi) &
            )
            ! Rotate stencil so the top right corner is always 1 and the bottom left corner always 0
            counter = 1
            if (rotation == 7) then
              do j_loc=1,stz
                do i_loc=1,stz
                  ml_tmp(counter) = ml_last_layer_output(i_loc + (j_loc-1)*stz) ! 0
                  counter = counter+1
                end do
              end do
            else if (rotation == 6) then
              do j_loc=1,stz
                do i_loc=1,stz
                  ml_tmp(counter) = ml_last_layer_output(j_loc + (i_loc -1)*stz) ! 0 flip
                  counter = counter+1
                end do
              end do
            else if (rotation == 5) then
              do j_loc=1,stz
                do i_loc=1,stz
                  ml_tmp(counter) = ml_last_layer_output((i_loc-1)*stz + (stz+1-j_loc)) ! 90
                  counter = counter+1
                end do
              end do
            else if (rotation == 4) then
              do j_loc=1,stz
                do i_loc=1,stz
                  ml_tmp(counter) = ml_last_layer_output((j_loc-1)*stz + (stz+1-i_loc)) ! 90 flip
                  counter = counter+1
                end do
              end do
            else if (rotation == 3) then
              do j_loc=1,stz
                do i_loc=1,stz
                  ml_tmp(counter) = ml_last_layer_output(stz*stz+1-i_loc - (j_loc-1)*stz) ! 180
                  counter = counter+1
                end do
              end do
            else if (rotation == 2) then
              do j_loc=1,stz
                do i_loc=1,stz
                  ml_tmp(counter) = ml_last_layer_output(stz*stz+1-j_loc-(i_loc-1)*stz) ! 180 flip
                  counter = counter+1
                end do
              end do
            else if (rotation == 1) then
              do j_loc=1,stz
                do i_loc=1,stz
                  ml_tmp(counter) = ml_last_layer_output((stz+1-i_loc)*stz+(j_loc-stz)) ! 270
                  counter = counter+1
                end do
              end do
            else if (rotation == 0) then
              do j_loc=1,stz
                do i_loc=1,stz
                  ml_tmp(counter) = ml_last_layer_output((stz+1-j_loc)*stz+(i_loc-stz)) ! 270 flip
                  counter = counter+1
                end do
              end do
            end if

            ! Overwrite NN features with rotated stencils
            do i_loc=1,stz**2
              ml_last_layer_output(i_loc) = ml_tmp(i_loc)
            end do

            deallocate(ml_tmp)
          end if
 
          if (1 == 1) then ! Edge2 switch         
            allocate(ml_stencil(stz, stz))

            ! Set everything to 0
            do j_loc=1,stz
              do i_loc=1,stz
                ml_stencil(j_loc, i_loc) = 0d0
              end do
            end do
            ! Write 1D ml_last_layer_output into 2D ml_stencil
            do j_loc=1,stz
              do i_loc=1,stz
                ml_stencil(j_loc, i_loc) = ml_last_layer_output((j_loc)*stz+(i_loc-stz))
              end do
            end do

            ! Print Test stencil
            ! print*, " " 
            ! print*, "Stencil Rotiert"
            ! do j_loc=1, stz
              ! print"(f6.3, f6.3, f6.3, f6.3, f6.3, f6.3, f6.3)", &
                ! ml_stencil(j_loc, 1), & 
                ! ml_stencil(j_loc, 2), & 
                ! ml_stencil(j_loc, 3), & 
                ! ml_stencil(j_loc, 4), & 
                ! ml_stencil(j_loc, 5), & 
                ! ml_stencil(j_loc, 6), & 
                ! ml_stencil(j_loc, 7)
            ! end do
            ! print*, " " 

            ! Reallocate ml_last_layer_output, because output of ml_edge will have length 2*stz**2 instead of stz**2
            deallocate(ml_last_layer_output)
            allocate(ml_last_layer_output(ml_n_features))

            ! X/Y seperated
            allocate(ml_interpolated_dir(ml_n_features/2))
            do i_loc=1,ml_n_features/2
              ml_interpolated_dir(i_loc) = 0d0
            end do
            ! Interpolate in y-direction and write results into first half of ml_last_layer_output
            call ml_edge(ml_stencil, 1, ml_interpolated_dir)
            counter = 1
            do i_loc=1,ml_n_features/2
              ml_last_layer_output(counter) = ml_interpolated_dir(i_loc)
              counter = counter + 1
            end do
            do i_loc=1,ml_n_features/2
              ml_interpolated_dir(i_loc) = 0d0
            end do
            ! Interpolate in x-direction and write results into second half of ml_last_layer_output
            call ml_edge(ml_stencil, 2, ml_interpolated_dir)
            do i_loc=1,ml_n_features/2
              ml_last_layer_output(counter) = ml_interpolated_dir(i_loc)
              counter = counter + 1
            end do
            deallocate(ml_stencil)
            deallocate(ml_interpolated_dir)

            ! X/Y together
            ! allocate(ml_interpolated_dir(ml_n_features))
            ! do i_loc=1,ml_n_features
              ! ml_interpolated_dir(i_loc) = 0d0
            ! end do
            ! ! Interpolate in y-direction and write results into first half of ml_last_layer_output
            ! call ml_edge(ml_stencil, 3, ml_interpolated_dir)
            ! counter = 1
            ! do i_loc=1,ml_n_features
              ! ml_last_layer_output(counter) = ml_interpolated_dir(i_loc)
              ! counter = counter + 1
            ! end do
            ! deallocate(ml_stencil)
            ! deallocate(ml_interpolated_dir)

          end if

          ! Test output
          ! print*, "Output 1:stz"
          ! do j_loc=1, stz
            ! print"(f6.3, f6.3, f6.3, f6.3, f6.3, f6.3, f6.3)", &
              ! ml_last_layer_output((j_loc-1)*stz+1), & 
              ! ml_last_layer_output((j_loc-1)*stz+2), & 
              ! ml_last_layer_output((j_loc-1)*stz+3), & 
              ! ml_last_layer_output((j_loc-1)*stz+4), & 
              ! ml_last_layer_output((j_loc-1)*stz+5), & 
              ! ml_last_layer_output((j_loc-1)*stz+6), & 
              ! ml_last_layer_output((j_loc-1)*stz+7)
          ! end do
          ! print*, "Output stz:2stz"
          ! do j_loc=stz+1, 2*stz
            ! print"(f6.3, f6.3, f6.3, f6.3, f6.3, f6.3, f6.3)", &
              ! ml_last_layer_output((j_loc-1)*stz+1), & 
              ! ml_last_layer_output((j_loc-1)*stz+2), & 
              ! ml_last_layer_output((j_loc-1)*stz+3), & 
              ! ml_last_layer_output((j_loc-1)*stz+4), & 
              ! ml_last_layer_output((j_loc-1)*stz+5), & 
              ! ml_last_layer_output((j_loc-1)*stz+6), & 
              ! ml_last_layer_output((j_loc-1)*stz+7)
          ! end do
          ! do i_loc=1,2*stz*(stz)
            ! print"(f6.3)", ml_last_layer_output(i_loc)
          ! end do
          ! STOP

          ! Export c/cm
          if (export == 1) then
            do j_loc=-stzh,stzh
              do i_loc=-stzh,stzh
                print*, cm(ha(i+i_loc, j+j_loc, 1))
              end do
            end do
            print*, 'cm'
            do i_loc=1,2*((stz-2)**2)  ! export rotated cm
              print*, ml_last_layer_output(i_loc)
            end do
          end if

          if (1 == 1) then ! matlab: y = (x-x_offset)*gain + y_min
	    ! Normalization
            do counter=1,ml_layer_nodes(1)
              ml_last_layer_output(counter) = (2d0*ml_last_layer_output(counter) - 1d0)
            end do
          end if

          if (timing == 1) then
            delta_pre = delta_pre + secnds(t2)
            t3 = secnds(0.0)
          end if
          ! Hidden Layers
          do n=2,ml_n_layers
            ! # ml_layer_input will be the input of the current layer, ml_last_layer_ouput will be used to save the results for the next layer
            ! Set ml_layer_input = ml_last_layer_output
            ! DAS IST WAHRSCHEINLICH LANGSAM
            allocate(ml_layer_input(size(ml_last_layer_output)))
            do counter=1,size(ml_last_layer_output)
              ml_layer_input(counter) = 0d0
            end do
            do counter=1,size(ml_layer_input)
              ml_layer_input(counter)=ml_last_layer_output(counter)
            end do
            ! Reallocate ml_last_layer_output to be filled with values of current layer
            deallocate(ml_last_layer_output)
            allocate(ml_last_layer_output(ml_layer_nodes(n)))
            do counter=1,ml_layer_nodes(n)
              ml_last_layer_output(counter) = 0d0
            end do

            ! Get index for weights and bias vector where current layer starts by summing length of weights of all previous layers
            w_idx_beg = 0d0
            do counter1=1,(n-1)
              w_idx_beg = w_idx_beg+ml_n_layerweights(counter1)
            end do
            if (ml_bias_on == 1) then
              b_idx_beg = 0d0
              do counter1=1,(n-1)
                b_idx_beg = b_idx_beg+ml_n_biasweights(counter1)
              end do
            end if

            if (timing == 1) then
              t4 = secnds(0.0)
            end if
            ! Calculate current layer input*weights + bias
            do o=1,ml_layer_nodes(n)
              ml_sum = 0d0
              ml_weight = 0d0
              ! Sum all connections (output of last layer l *weight k l) for node k of current layer
              do p=1,ml_layer_nodes(n-1)
                ! Get weight of connection between last layer and current layer
                ml_weight = ml_layerweights(w_idx_beg+(p-1)*ml_layer_nodes(n)+o)
                ! Add influence of that connection
                ml_sum = ml_sum+ml_layer_input(p)*ml_weight
              end do 
              if (ml_bias_on == 1) then
                ml_biasweight = ml_biasweights(b_idx_beg+o)
                ml_last_layer_output(o) = ml_sum+ml_biasweight
              else
                ml_last_layer_output(o) = ml_sum
              end if
            end do
            if (timing == 1) then
              delta_tmp = delta_tmp + secnds(t4)
            end if

            ! Apply activation function (only for relu, do nothing if activation is linear)
            if (ml_layer_activation(n) == 'relu') then
              do o=1,ml_layer_nodes(n)
                ! Set every output that is smaller then 0 to 0
                if (ml_last_layer_output(o) < 0d0) then
                  ml_last_layer_output(o) = 0d0
                end if
              end do
            else if (ml_layer_activation(n) == 'tanh') then
              do o=1,ml_layer_nodes(n)
                ! Set every output that is smaller then 0 to 0
                ml_last_layer_output(o) = 2d0/(1d0+exp(-2d0*ml_last_layer_output(o)))-1d0
              end do
            end if

            ! Deallocate input for usage in next layer
            deallocate(ml_layer_input)
          end do

          if (1 == 1) then ! matlab: y = (y-y_min)/gain + offset
	    ! Normalization
            ml_last_layer_output(1) = 1d0*(ml_last_layer_output(1)*(4d0/10d0-(-4d0/10d0))+(4d0/10d0+(-4d0/10d0)))*0.5d0
          end if

          ! Last output of ML network is the curvature 
	  ! Divide by domain size:
          ! dscurv2(inp)=ml_last_layer_output(1)*0.5d0*gridsize/8d0 ! Static Bubble
          dscurv2(inp)=-1d0*ml_last_layer_output(1)*0.5d0*gridsize/0.075d0 ! Osc. Bubble
          ! dscurv2(inp)=ml_last_layer_output(1)*0.5d0*gridsize/1d0 ! Rising Bubble
          ! dscurv2(inp)=ml_last_layer_output(1)*gridsize/0.075d0 ! Osc. Bubble CVOFLS

          if (export == 1) then
            ! print*, 'last output', dscurv2(inp)
            print*, 'last output', ml_last_layer_output(1)
            !print*, 'last output', rotation
          endif
          ! STOP
          deallocate(ml_last_layer_output)
          if (timing == 1) then
            delta_ml = delta_ml + secnds(t3)
          end if
        else
          ! If concentration in stencil midpoint is 0 or 1, set curvature to 0
          dscurv2(inp)=0d0
        end if
      end do
    end do
  end do
end do
! print*, 'STOP'
! STOP
if (timing == 1) then
  delta = secnds(t1)
  print*, 'Pre-Processing+Auswertung'
  print*, delta
  print*, 'Pre-Processing'
  print*, delta_pre
  print*, 'Ausmultiplizieren des Netzes'
  print*, delta_tmp
  print*, 'Gesamte Auswertung des Netzes'
  print*, delta_ml
  t3 = secnds(0.0)
end if

curvfiltering=.true.  ! war false
 !1-lsp0,2-binominal,3-5pointcheby,4-lsp2
curvfilter=3
curvweightening=.true.
weightsexp=1

! Filter the curvature afterwards
if (curvfiltering .eqv. .true.) then

  !first step
  call setzero(weights,nxyza,1,nxyza,ier)
  call setzero(tempcurv,nxyza,1,nxyza,ier)
  call exall(ngr,dscurv2,1,1, &
#            include "listexall.h"
             ier)
  do icphi=0,mcphi
    incst=icphi*nxyza
    incst1=incst+1
    call exall(ngr,cm(incst1),1,1, &
#            include "listexall.h"
             ier)
  end do
  do m=1,nblo
    call setind(ngr,m, &
#               include "listsetind.h"
                ier)
    if (thirddim==1) then
      kstart=1
      kend=nk
    else
      kstart=2
      kend=2
    end if
    do k=kstart,kend
      do i=1,ni
        do j=1,nj
          inp=ha(i,j,k)
          ! more weight to cells with 0.5 concentration, less close to 0 or 1
          if (dscurv2(inp)/=0d0) then
            weights(inp)=(1-2d0*abs(0.5d0-cm(inp)))**(weightsexp)
          else
            weights(inp)=0d0
          end if
        end do
      end do
    end do
  end do
  do m=1,nblo
    call setind(ngr,m, &
#               include "listsetind.h"
                ier)
    do k=2,nkm
      do i=2,nim
        do j=2,njm
          inp=ha(i,j,k)
          if (dscurv2(inp)/=0d0) then
            if (thirddim==0) then
              tempcurv(inp)=(dscurv2(inp)*weights(inp) + &
                dscurv2(ha(i+1,j,k))*weights(ha(i+1,j,k))+ &
                dscurv2(ha(i-1,j,k))*weights(ha(i-1,j,k))+ &
                dscurv2(ha(i,j+1,k))*weights(ha(i,j+1,k))+ &
                dscurv2(ha(i,j-1,k))*weights(ha(i,j-1,k)))/ &
                (weights(inp)+weights(ha(i+1,j,k))+weights(ha(i-1,j,k))+weights(ha(i,j+1,k))+weights(ha(i,j-1,k)))
            else
              tempcurv(inp)=(dscurv2(inp)*weights(inp) + &
                dscurv2(ha(i+1,j,k))*weights(ha(i+1,j,k))+ &
                dscurv2(ha(i-1,j,k))*weights(ha(i-1,j,k))+ &
                dscurv2(ha(i,j+1,k))*weights(ha(i,j+1,k))+ &
                dscurv2(ha(i,j-1,k))*weights(ha(i,j-1,k))+ &
                dscurv2(ha(i,j,k+1))*weights(ha(i,j,k+1))+ &
                dscurv2(ha(i,j,k-1))*weights(ha(i,j,k-1)))/ &
                (weights(inp)+weights(ha(i+1,j,k))+weights(ha(i-1,j,k))+weights(ha(i,j+1,k))+weights(ha(i,j-1,k)) &
                +weights(ha(i,j,k+1))+weights(ha(i,j,k-1)))
            end if
          end if
        end do
      end do
    end do
  end do
  do m=1,nblo
    call setind(ngr,m, &
#               include "listsetind.h"
                ier)
    do k=2,nkm
      do i=2,nim
        do j=2,njm
          inp=ha(i,j,k)
          if (tempcurv(inp)/=0d0) then
            dscurv2(inp)=tempcurv(inp)
          end if
        end do
      end do
    end do
  end do
end if
! Weightening of curvature
if (curvweightening .eqv. .true.) then

  !first step
  call setzero(weights,nxyza,1,nxyza,ier)
  call setzero(tempcurv,nxyza,1,nxyza,ier)
  call exall(ngr,dscurv2,1,1, &
#            include "listexall.h"
             ier)
  do icphi=0,mcphi
    incst=icphi*nxyza
    incst1=incst+1
    call exall(ngr,cm(incst1),1,1, &
#            include "listexall.h"
             ier)
  end do
  do m=1,nblo
    call setind(ngr,m, &
#               include "listsetind.h"
                ier)
    if (thirddim==1) then
      kstart=1
      kend=nk
    else
      kstart=2
      kend=2
    end if
    do k=kstart,kend
      do i=1,ni
        do j=1,nj
          inp=ha(i,j,k)
          ! more weight to cells with 0.5 concentration, less close to 0 or 1
          if (dscurv2(inp)/=0d0) then
            weights(inp)=1d0
          else
            weights(inp)=0d0
          end if
        end do
      end do
    end do
  end do
  do m=1,nblo
    call setind(ngr,m, &
#               include "listsetind.h"
                ier)
    do k=2,nkm
      do i=2,nim
        do j=2,njm
          inp=ha(i,j,k)
          ! if (dscurv2(inp)/=0d0) then
          if ((cm(inp) > 0.0075d0) .AND. (cm(inp) < 0.9925d0)) then
            if (thirddim==0) then
              tempcurv(inp)=(dscurv2(inp)*weights(inp) + &
                dscurv2(ha(i+1,j,k))*weights(ha(i+1,j,k))+ &
                dscurv2(ha(i-1,j,k))*weights(ha(i-1,j,k))+ &
                dscurv2(ha(i,j+1,k))*weights(ha(i,j+1,k))+ &
                dscurv2(ha(i,j-1,k))*weights(ha(i,j-1,k)))/ &
                (weights(inp)+weights(ha(i+1,j,k))+weights(ha(i-1,j,k))+weights(ha(i,j+1,k))+weights(ha(i,j-1,k)))
            else
              tempcurv(inp)=(dscurv2(inp)*weights(inp) + &
                dscurv2(ha(i+1,j,k))*weights(ha(i+1,j,k))+ &
                dscurv2(ha(i-1,j,k))*weights(ha(i-1,j,k))+ &
                dscurv2(ha(i,j+1,k))*weights(ha(i,j+1,k))+ &
                dscurv2(ha(i,j-1,k))*weights(ha(i,j-1,k))+ &
                dscurv2(ha(i,j,k+1))*weights(ha(i,j,k+1))+ &
                dscurv2(ha(i,j,k-1))*weights(ha(i,j,k-1)))/ &
                (weights(inp)+weights(ha(i+1,j,k))+weights(ha(i-1,j,k))+weights(ha(i,j+1,k))+weights(ha(i,j-1,k)) &
                +weights(ha(i,j,k+1))+weights(ha(i,j,k-1)))
            end if
          end if
        end do
      end do
    end do
  end do
  do m=1,nblo
    call setind(ngr,m, &
#               include "listsetind.h"
                ier)
    do k=2,nkm
      do i=2,nim
        do j=2,njm
          inp=ha(i,j,k)
          if ((tempcurv(inp)/=0d0)) then
            dscurv2(inp)=tempcurv(inp)
          end if
        end do
      end do
    end do
  end do
end if

! Relaxation when changes are big
if (0 == 1) then
  if (it .ne. ml_last_it) then
    do m=1,nblo
      call setind(ngr,m, &
#             include "listsetind.h"
                 ier)
      do k=2,nkm
        do j=2,njm
          do i=2,nim
            ! Overwrite last curvature field
            ml_last_curv(i-1, j-1) = ml_tmp_curv(i-1, j-1)
          end do
        end do
      end do
    end do
  end if
  do m=1,nblo
    call setind(ngr,m, &
#             include "listsetind.h"
               ier)
    do k=2,nkm
      do j=2,njm
        do i=2,nim
          inp=ha(i,j,k)
          ! Compare with last curvature field
          if ((ml_last_curv(i-1, j-1) .ne. 0) .and. (dscurv2(inp) .ne. 0)) then  ! only continue if last curvature was not 0
            ! If new value is more than 50% different from old value
            if (abs(ml_last_curv(i-1, j-1) - dscurv2(inp)) > abs(ml_last_curv(i-1, j-1)/2d0)) then
              ! Relax value
              dscurv2(inp) = 0.1*dscurv2(inp) + 0.9*ml_last_curv(i-1, j-1)
            end if
          else if (dscurv2(inp) .ne. 0) then
            ml_sum = 0d0
            ml_loc = 0d0
            ! Get mean of neighbours ne 0
            if (ml_last_curv(i, j-1) .ne. 0d0) then
              ml_loc = ml_loc + 1d0
              ml_sum = ml_sum + ml_last_curv(i, j-1)
            end if
            if (ml_last_curv(i-1, j) .ne. 0d0) then
              ml_loc = ml_loc + 1d0
              ml_sum = ml_sum + ml_last_curv(i-1, j)
            end if
            if (ml_last_curv(i-1, j-2) .ne. 0d0) then
              ml_loc = ml_loc + 1d0
              ml_sum = ml_sum + ml_last_curv(i-1, j-2)
            end if
            if (ml_last_curv(i-2, j-1) .ne. 0d0) then
              ml_loc = ml_loc + 1d0
              ml_sum = ml_sum + ml_last_curv(i-2, j-1)
            end if
            if (ml_loc .ne. 0d0) then
              ml_sum = ml_sum/ml_loc
            else 
              ml_sum = 0d0
            end if
            if ((abs(ml_sum - dscurv2(inp)) > abs(ml_sum/2d0)) .and. (ml_sum .ne. 0d0)) then
              ! Relax value
              dscurv2(inp) = 0.1*dscurv2(inp) + 0.9*ml_sum
            end if
          end if
          ! Store last curvature field
          ml_tmp_curv(i-1, j-1) = dscurv2(inp)
        end do
      end do
    end do
  end do
  ml_last_it = it
end if
if (timing == 1) then
  print*, 'Post-Processing'
  delta = secnds(t3)
  print*, delta
  print*, 'Gesamt'
  delta = secnds(t1)
  print*, delta
  print*, ' '
end if

! delta = secnds(t1)
! print*, delta
end subroutine curv_ml

