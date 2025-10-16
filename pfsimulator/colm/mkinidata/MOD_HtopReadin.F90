#include <define.h>

MODULE MOD_HtopReadin

   USE MOD_Precision
   IMPLICIT NONE
   SAVE

   ! PUBLIC MEMBER FUNCTIONS:
   PUBLIC :: HTOP_readin

CONTAINS

   SUBROUTINE HTOP_readin (dir_landdata, lc_year, numpatch)

! ===========================================================
! Read in the canopy tree top height
! ===========================================================

   USE MOD_Precision
   USE MOD_SPMD_Task
   USE MOD_Vars_Global
   USE MOD_Const_LC
   USE MOD_Const_PFT
   USE MOD_Vars_TimeInvariants
   !USE MOD_LandPatch
#if (defined LULC_IGBP_PFT || defined LULC_IGBP_PC)
   USE MOD_LandPFT
   USE MOD_Vars_PFTimeInvariants
   USE MOD_Vars_PFTimeVariables
#endif
   !USE MOD_NetCDFVector
#ifdef SinglePoint
   USE MOD_SingleSrfdata
#endif

   IMPLICIT NONE

   integer, intent(in) :: lc_year    ! which year of land cover data used
   character(len=256), intent(in) :: dir_landdata

   ! Local Variables
   character(len=256) :: c
   character(len=256) :: landdir, lndname, cyear
   integer :: i,j,t,p,ps,pe,m,n,npatch

   real(r8), allocatable :: htoplc  (:)
   real(r8), allocatable :: htoppft (:)
   integer, intent(in) :: numpatch

      write(cyear,'(i4.4)') lc_year
      landdir = trim(dir_landdata) // '/htop/' // trim(cyear)


#ifdef LULC_USGS

      IF (p_is_worker) THEN
         DO npatch = 1, numpatch
            m = patchclass(npatch)

            htop(npatch) = htop0(m)
            hbot(npatch) = hbot0(m)

         ENDDO
      ENDIF

#endif

#ifdef LULC_IGBP
#ifdef SinglePoint
      allocate (htoplc (numpatch))
      htoplc(:) = SITE_htop
#else
      lndname = trim(landdir)//'/htop_patches.nc'
      !CALL ncio_read_vector (lndname, 'htop_patches', landpatch, htoplc)
      allocate (htoplc (numpatch))
#endif

      IF (p_is_worker) THEN
         DO npatch = 1, numpatch
            m = patchclass(npatch)

            htop(npatch) = htop0(m)
            hbot(npatch) = hbot0(m)

            ! trees or woody savannas
            IF ( m<6 .or. m==8) THEN
               ! 01/06/2020, yuan: adjust htop reading
               !IF (htoplc(npatch) > 2.) THEN
               !   htop(npatch) = htoplc(npatch)
               !   hbot(npatch) = htoplc(npatch)*hbot0(m)/htop0(m)
               !   hbot(npatch) = max(1., hbot(npatch))
               !   !htop(npatch) = max(htop(npatch), hbot0(m)*1.2)
               !ENDIF
            ENDIF

         ENDDO
      ENDIF

      IF (allocated(htoplc))   deallocate ( htoplc )
#endif


#if (defined LULC_IGBP_PFT || defined LULC_IGBP_PC)
#ifdef SinglePoint
      allocate(htoppft(numpft))
      htoppft = pack(SITE_htop_pfts, SITE_pctpfts > 0.)
#else
      lndname = trim(landdir)//'/htop_pfts.nc'
      CALL ncio_read_vector (lndname, 'htop_pfts', landpft,   htoppft)
#endif

      IF (p_is_worker) THEN
         DO npatch = 1, numpatch
            t = patchtype(npatch)
            m = patchclass(npatch)

            IF (t == 0) THEN
               ps = patch_pft_s(npatch)
               pe = patch_pft_e(npatch)

               DO p = ps, pe
                  n = pftclass(p)

                  htop_p(p) = htop0_p(n)
                  hbot_p(p) = hbot0_p(n)

                  ! for trees
                  ! 01/06/2020, yuan: adjust htop reading
                  IF ( n>0 .and. n<9 .and. htoppft(p)>2.) THEN
                     htop_p(p) = htoppft(p)
                     hbot_p(p) = htoppft(p)*hbot0_p(n)/htop0_p(n)
                     hbot_p(p) = max(1., hbot_p(p))
                  ENDIF
               ENDDO

               htop(npatch) = sum(htop_p(ps:pe)*pftfrac(ps:pe))
               hbot(npatch) = sum(hbot_p(ps:pe)*pftfrac(ps:pe))

            ELSE
               htop(npatch) = htop0(m)
               hbot(npatch) = hbot0(m)
            ENDIF

         ENDDO
      ENDIF

      IF (allocated(htoppft)) deallocate(htoppft)
#endif

   END SUBROUTINE HTOP_readin

END MODULE MOD_HtopReadin
