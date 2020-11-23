#include "listdefines.h"
!#####################################################################
real*8 function fden(press,tmpr,iflr,sumcm)
!#####################################################################
! Density depends in general on temperature, pressure, and the fluid which is modeled.
! The fluid can change only by the 'flow region' (iflr). Please note that 'tmpr' is the absolute
! temperature whereas 'press' is only the relative pressure. sumcm is the recursive mean mol mass.
!#####################################################################
implicit none

! dummy variables
real*8, intent(in)  :: press,tmpr,sumcm
integer, intent(in) :: iflr
!#####################################################################

fden=1.d0

! Water
!fden=-1359.6466d0+4.4739281d0*tmpr-0.0042091357d0*tmpr*tmpr+635713.25d0/tmpr-65348666.d0/tmpr**2

! Air
!fden=0.96737341d0-0.00028655237d0*tmpr-1463.8211d0/tmpr+46843.588d0/t**1.5-347974.74d0/tmpr**2

! Glass
!fden=2700.d0

! Silicon melt (T=1703 K)
!fden=2570.d0

return
end function fden

!#####################################################################
real*8 function fden2(inp,press,tmpr,iflr,sumcm,c)
!#####################################################################
! Called only in Multiphase calculation 
! Function for calculation of local material properties (density)
!#####################################################################
implicit none

#include "cb3dall.h"
#include "cbglobaldim.h"

integer, intent(in) :: inp,iflr
real*8, intent(in) :: press,tmpr,sumcm,c(*)
! dummy variables
integer :: icphi,incst
real*8 :: denmp(2)
!#####################################################################

! First Fluid
denmp(1)=787.88d0
! Second Fluid
denmp(2)=1.1768d0 !1.2d0 !1.205d0

! Calculation of local density
fden2=0.d0
do icphi=0,mcphi
  incst=icphi*nxyza 
  fden2=fden2+c(inp+incst)*denmp(1+icphi)
end do

return
end function fden2


!#####################################################################
real*8 function fvis(tmpr,iflr)
!#####################################################################
! It is assumed that dynamic viscosity depends only on the temperature and the fluid which is modeled
! (selected by 'iflr'). The unit of the dynamic viscosity is [kg/m/s].
! Note: For 'fvis => 1' the corresponding flow region is assumed to consist of solid blocks only and all
! velocities and turbulent quantities are set to zero.
!#####################################################################
implicit none

! dummy variables
real*8, intent(in)  :: tmpr
integer, intent(in) :: iflr
!#####################################################################

fvis=1.d0

! Water
!fvis=1.d0/(-1395.1278d0+9.537673d-5*tmpr**3)

! Air
!fvis=-9.2014393d-6+4.4397544d-9*tmpr-1.4046401d-14*tmpr**2.5+3.2787496d-16*tmpr**3+1.5276332d-6*sqrt(tmpr)
! At 293.15 degrees Kelvin and a pressure of 1 Bar
!fvis=1.806d-5

! Glass
!fvis=1.d0

! Silicon melt
!fvis=7.750d-04

return
end function fvis


!#####################################################################
      real*8 function fvis2(inp,tmpr,iflr,c)
!#####################################################################
! Called only in Multiphase calculation 
! Function for calculation of local material properties (viscosity)
!#####################################################################
implicit none

#include "cb3dall.h"
#include "cbglobaldim.h"

integer, intent(in) :: inp,iflr
real*8, intent(in) :: tmpr,c(*)
! dummy variables
real*8 :: vismp(2)
integer :: icphi,incst
!#####################################################################

! Fluid 1
vismp(1)=2.4d-2

! Fluid 2
vismp(2)=2.0d-3

! Calculation of local viscosity
fvis2=0.d0
do icphi=0,mcphi
  incst=icphi*nxyza 
  fvis2=fvis2+c(inp+incst)*vismp(1+icphi)
end do

return
end function fvis2


!#####################################################################
real*8 function rgas(nfr)
!#####################################################################
! Gas constant
!#####################################################################
implicit none

! dummy variables
integer, intent(in) :: nfr
!#####################################################################

! Air
rgas=287.1d0

return
end function rgas


!#####################################################################
real*8 function kappa(nfr)
!#####################################################################
! Isotropic exponent
!#####################################################################
implicit none

! dummy variables
integer, intent(in) :: nfr
!#####################################################################

! ideal gas
kappa=1.67

! air
!kappa=1.4

return
end function kappa


!#####################################################################
real*8 function flam(tmpr,iflr)
!#####################################################################
! It is assumed that the heat conductivity depends only on the temperature and the fluid that is modeled (selected by 'iflr').
! The unit of heat conductivity is [W/m**2/K].
!#####################################################################
implicit none

! dummy variables
real*8, intent(in)  :: tmpr
integer, intent(in) :: iflr
!#####################################################################

! Water
!flam=sqrt(0.63384432d0-23521.208d0/tmpr**2)
! At 293.15 degrees Kelvin
flam=0.6001176d0

! Air
!flam=0.042491107d0+1.4835066d-6*tmpr**1.5-8.5222871d-12*tmpr**3-0.41142788d0/sqrt(tmpr)
! At 293.15 degrees Kelvin
!flam=0.025d0

! Glass
!flam=220.d0

! Silicon melt (T=1685 K)
!flam=66.9d0

return
end function flam


!#####################################################################
real*8 function fcp(tmpr,iflr)
!#####################################################################
! It is assumed that the heat capacity depends only on the temperature and the fluid that is modeled (selected by 'iflr').
! The unit of heat capacity is [J/kg/K].
!#####################################################################
implicit none

real*8, intent(in)  :: tmpr
integer, intent(in) :: iflr
!#####################################################################

! Water
!fcp=-1.703933d8-1083.3513d0*tmpr+744995.96d0*(log(tmpr))**2+1.143102d9/log(tmpr)-9.3402192d8/sqrt(tmpr)
! At 293.15 degrees Kelvin
fcp=4187.d0

! Air
!fcp=-0.014860411d0+8.3859987d-10*tmpr**2.5-1.7304405d-11*tmpr**3+0.002323523d0*sqrt(tmpr)

! Glass
!fcp=840.d0

! Silicon melt (T=1685)
!fcp=915.d0

return
end function fcp


!#####################################################################
real*8 function fbeta(tmpr,iflr)
!#####################################################################
! It is assumed that the coefficient of thermal expansion depends only on the temperature and the modeled fluid
! (selected by 'iflr').
!#####################################################################
implicit none

! dummy variables
real*8, intent(in)  :: tmpr
integer, intent(in) :: iflr
!#####################################################################

fbeta=0.d0

! Water
!fbeta=0.2067d-3

! Air
!fbeta=3.421d-3

! Glass
!fbeta=?

! Silicon melt

!fbeta=1.4d-4

return
end function fbeta


!#####################################################################
subroutine fmolm(mcphi,gmol,ier)
!#####################################################################
! Mol mass of species i.
!#####################################################################
implicit none

! dummy variables
integer, intent(in) :: mcphi
real*8, intent(out) :: gmol(0:*)
integer :: ier
!#####################################################################

! H2, C2H6, C2H4

gmol(0)=2.016
gmol(1)=30.07
gmol(2)=28.054

return
end subroutine fmolm


!#####################################################################
real*8 function fdiff(t,iflr,icphi)
!#####################################################################
! Diffusion coefficient of species icphi.
!#####################################################################
implicit none

! dummy variables
real*8, intent(in)  :: t
integer, intent(in) :: iflr,icphi

! local variables
real*8  :: p0
!#####################################################################

p0=0.1

! C2H6 and C2H4 in H2

if (icphi == 1) fdiff=5.35e-5/p0*(t/298.15)**1.712
if (icphi == 2) fdiff=5.768e-5/p0*(t/298.15)**1.714

return
end function fdiff


!#####################################################################
real*8 function fdifft(t,iflr,icphi)
!#####################################################################
! Thermal diffusion coefficient of species icphi.
!#####################################################################
implicit none

! dummy variables
real*8, intent(in)  :: t
integer, intent(in) :: iflr,icphi
!#####################################################################

fdifft=0.

return
end function fdifft


!#####################################################################
real*8 function femiss(t,iflr)
!#####################################################################
! The emissivity of a medium depends only on the temperature of the surface, the wavelength of the light and the medium.
! femiss  = emissivity of the melt
! wavelen = Wavelength of light (in 'nm', range: 500 nm - 900 nm)
!#####################################################################
implicit none

! dummy variables
real*8, intent(in)  :: t
integer, intent(in) :: iflr

! local variables
real*8  wavelen
!#####################################################################

wavelen=700.d0

femiss=0.265d0-(5.d-5*wavelen)

return
end function femiss

!#####################################################################

real*8 function frks(x,y,z,iflr,igb,ifa)
!#####################################################################
! Wall roughness [m], applied on every patch.
! x, y, z are the coordinates of the center point of the current patch.
!#####################################################################
implicit none

! dummy variables
real*8, intent(in)  :: x,y,z
integer, intent(in) :: iflr,igb,ifa
!#####################################################################

! All walls are smooth:
frks = 0.d0

! All walls have the same roughness:
!frks = 0.001d0

! The walls have different roughnesses:

! - The walls have the same roughness in one flow region:
!frks = 0.d0
!if(iflr.eq.1) frks = 0.0005d0
!if(iflr.eq.2) frks = 0.001d0

! The patches are being addressed by their geometric block number and their direction (west, east,... ):
!frks = 0.d0
!if(igb.eq.5 .and. ifa.eq.3) frks = 0.001d0
!if(igb.eq.7 .and. ifa.eq.2) frks = 0.0002d0

return
end function frks


!#####################################################################
real*8 function fmixdiff(t,iflr,imphi)
!#####################################################################
! Diffusion coefficient for combustion module.
!#####################################################################
implicit none

! dummy variables
real*8, intent(in)  :: t
integer, intent(in) :: iflr,imphi

! local variables
real*8  :: p0
!#####################################################################

p0=0.1

if (imphi == 1) fmixdiff=0.0003d0
if (imphi == 2) fmixdiff=0.0003d0

return
end function fmixdiff


!#####################################################################
real*8 function fschmidtt()
!#####################################################################
! Turbulent Schmidt number.
!#####################################################################
implicit none
!#####################################################################

fschmidtt=0.9d0

return
end function fschmidtt

!#####################################################################
real*8 function fsos(tmpr,iflr,inp,c)
!#####################################################################
!Speed of sound
!#####################################################################
implicit none
integer iflr,inp
real*8 tmpr,c(*)
#include "cb3dall.h"
#include "cbglobaldim.h"
integer icphi,incst
real*8  sosmp(2)
!#####################################################################
! Speed of Sound for Phase 1 in m/s
	sosmp(1) = 1142.d0
! Speed of Sound for Phase 2 in m/s
	sosmp(2) = 343.d0
      
fsos=0.0d0
do icphi=0,mcphi
  incst=icphi*nxyza 
  fsos=fsos+c(inp+incst)*sosmp(1+icphi)
 end do

return
end function fsos

!#####################################################################
real*8 function acaddu(x_tmp,y_tmp,z_tmp)
!#####################################################################
! Additional background velocity for acoustics in x direction
!#####################################################################
implicit none

! dummy variables
integer :: iflr
real*8 :: tmpr,x_tmp,y_tmp,z_tmp
!#####################################################################
      
acaddu=0.d0

return
end function acaddu


!#####################################################################
real*8 function acaddv(x_tmp,y_tmp,z_tmp)
!#####################################################################
! Additional background velocity for acoustics in y direction
!#####################################################################
implicit none

! dummy variables
integer :: iflr
real*8 :: tmpr,x_tmp,y_tmp,z_tmp
!#####################################################################
      
acaddv=0.d0

return
end function acaddv


!#####################################################################
real*8 function acaddw(x_tmp,y_tmp,z_tmp)
!#####################################################################
! Additional background velocity for acoustics in z direction
!#####################################################################
implicit none

! dummy variables
integer :: iflr
real*8 :: tmpr,x_tmp,y_tmp,z_tmp
!#####################################################################
      
acaddw=0.d0

return
end function acaddw


