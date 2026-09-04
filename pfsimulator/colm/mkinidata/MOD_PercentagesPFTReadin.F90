#include <define.h>

MODULE MOD_PercentagesPFTReadin

!-----------------------------------------------------------------------
   USE MOD_Precision
   IMPLICIT NONE
   SAVE

! PUBLIC MEMBER FUNCTIONS:
   PUBLIC :: pct_readin

CONTAINS

   SUBROUTINE pct_readin (dir_landdata, lc_year)

   USE MOD_Precision
   USE MOD_Vars_Global
   USE MOD_SPMD_Task
   USE MOD_NetCDFVector
   USE MOD_LandPatch
#ifdef CROP
   USE MOD_LandCrop
#endif
#ifdef RangeCheck
   USE MOD_RangeCheck
#endif
#if (defined LULC_IGBP_PFT || defined LULC_IGBP_PC)
   USE MOD_LandPFT
   USE MOD_Vars_PFTimeInvariants
#endif
#ifdef SinglePoint
   USE MOD_SingleSrfdata
#endif
   IMPLICIT NONE

   integer, intent(in) :: lc_year
   character(len=256), intent(in) :: dir_landdata
   ! Local Variables
   character(len=256) :: lndname, cyear
   real(r8), allocatable :: sumpct (:)
   integer :: npatch, ipatch

      write(cyear,'(i4.4)') lc_year
#if (defined LULC_IGBP_PFT || defined LULC_IGBP_PC)
#ifndef SinglePoint
      lndname = trim(dir_landdata)//'/pctpft/'//trim(cyear)//'/pct_pfts.nc'
      CALL ncio_read_vector (lndname, 'pct_pfts', landpft, pftfrac)
#else
      pftfrac = pack(SITE_pctpfts, SITE_pctpfts > 0.)
#endif

#if (defined CROP)
#ifndef SinglePoint
      lndname = trim(dir_landdata)//'/pctpft/'//trim(cyear)//'/pct_crops.nc'
      CALL ncio_read_vector (lndname, 'pct_crops', landpatch, cropfrac)
#else
      IF (SITE_landtype == CROPLAND) THEN
         cropfrac = pack(SITE_pctcrop, SITE_pctcrop > 0.)
      ELSE
         cropfrac = 0.
      ENDIF
#endif
#endif

#ifdef RangeCheck
      IF (p_is_worker) THEN
         npatch = count(patchtypes(landpatch%settyp) == 0)
         allocate (sumpct (npatch))

         npatch = 0
         DO ipatch = 1, numpatch
            IF (patchtypes(landpatch%settyp(ipatch)) == 0) THEN
               npatch = npatch + 1
               sumpct(npatch) = sum(pftfrac(patch_pft_s(ipatch):patch_pft_e(ipatch)))
            ENDIF
         ENDDO

      ENDIF

      CALL check_vector_data ('Sum PFT pct', sumpct)
#if (defined CROP)
      CALL check_vector_data ('CROP pct', cropfrac)
#endif
#endif

#endif

      IF (allocated(sumpct)) deallocate(sumpct)

   END SUBROUTINE pct_readin

END MODULE MOD_PercentagesPFTReadin
