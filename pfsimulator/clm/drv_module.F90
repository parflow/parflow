!#include <misc.h>

module drv_module 
!=========================================================================
!
!  CLMCLMCLMCLMCLMCLMCLMCLMCL  A community developed and sponsored, freely   
!  L                        M  available land surface process model.  
!  M --COMMON LAND MODEL--  C  
!  C                        L  CLM WEB INFO: http://clm.gsfc.nasa.gov
!  LMCLMCLMCLMCLMCLMCLMCLMCLM  CLM ListServ/Mailing List: 
!
!=========================================================================
! drv_module.f90: 
!
! DESCRIPTION:
!  Module for 1-D land model driver variable specification.
!
! REVISION HISTORY:
!  15 Jan 2000: Paul Houser; Initial code
!   3 Mar 2000: Jon Radakovich; Revision for diagnostic output
!=========================================================================     
! $Id: drv_module.F90,v 1.1.1.1 2006/02/14 23:05:52 kollet Exp $
!=========================================================================

  use precision
  implicit none

  public drvdec
  type drvdec

!=== Driver User-Defined Parameters ======================================
     real(r8) ::               &
          mina,             & !Min grid area for tile (%)
          udef                !Undefined value

     character*40 ::       &
          vegtf,            & !Vegetation Tile Specification File                   
          vegpf,            & !Vegetation Type Parameter Values
          poutf1d,          & !CLM 1D Parameter Output File
          metf1d,           & !Meterologic input file
          outf1d,           & !CLM output file
          rstf                !CLM active restart file

!=== CLM Parameters ======================================================
     integer :: nch          !actual number of tiles

!=== Driver Parameters ====================================================
     integer :: nc           !Number of Columns in Grid
     integer :: nr           !Number of Rows in Grid
     integer :: nt           !Number of Vegetation Types 
     integer :: startcode    !0=restart date, 1=card date
     integer :: sss          !Starting Second 
     integer :: sdoy         !Starting Day of Year 
     integer :: smn          !Starting Minute 
     integer :: shr          !Starting Hour 
     integer :: sda          !Starting Day 
     integer :: smo          !Starting Month 
     integer :: syr          !Starting Year  
     integer :: ess          !Ending Second
     integer :: emn          !Ending Minute
     integer :: edoy         !Ending Day of Year
     integer :: ehr          !Ending Hour
     integer :: eda          !Ending Day
     integer :: emo          !Ending Month
     integer :: eyr          !Ending Year
     integer :: ts           !Timestep (seconds) 
     integer :: ts_old
     real(r8) :: writeintc   !CLM Output Interval (hours)
     integer :: maxt         !Maximum tiles per grid  

!=== Timing Variables ==========
     real*8  :: time                  !CLM Current Model Time in Years
     real*8  :: etime                 !CLM End Time in Years
     integer :: pda                   !CLM Previous Timestep Day
     integer :: doy,yr,mo,da,hr,mn,ss !CLM Current Model Timing Variables   
     integer :: endtime               !CLM Stop (0=continue time looping)
     real(r8):: day,gmt,eday,egmt,sgmt

!=== Arguments ==========================================================
     real(r8) :: ctime                !CLM Restart Time 
     integer :: cyr,cmo,cda       !Restart Model Timing Variables
     integer :: chr,cmn,css       !Restart Model Timing Variables

!=== Initial CLM conditions =============================================
     real(r8) :: t_ini                !Initial temperature [K] 
     real(r8) :: h2osno_ini              !Initial snow cover, water equivalent [mm] 
     real(r8) :: sw_ini               !Initial average soil volumetric water&ice content [m3/m3] 

!=== CLM diagnostic parameters ==========================================
     integer :: surfind      !Number of surface diagnostic variables
     integer :: soilind      !Number of soil layer diagnostic variables
     integer :: snowind      !Number of snow layer diagnostic variables

     integer :: vclass            !Vegetation Classification Scheme (1=UMD,2=IGBP,etc.) NOT the index 
     integer :: clm_ic            !CLM Initial Condition Source
  
!=== CLM.PF varibales
     integer  :: sat_flag         ! 0: enough storage in the domain; 1: too little storage in the domain, full saturation
     real(r8) :: dx,dy,dz         
     real(r8) :: begwatb, endwatb ! beg and end water balance over domain      
     
!=== End Variable List ===================================================
  end type drvdec

end module drv_module
