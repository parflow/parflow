subroutine close_files (clm,drv)
 use clmtype             ! CLM tile variables
 use clm_varpar, only : nlevsoi ! Stefan: added because of flux array that is passed
 use precision
 use drv_module          ! 1-D Land Model Driver variables
 implicit none

 type (clm1d) :: clm(nlevsoi)
 type (drvdec):: drv              

 close(6)
 close(199)
  
 close(2000)
 close(2001)
 close(2002)
 close(2003)
 close(2004)
 close(2005)
 close(2006)
 close(2008)
  
 end subroutine close_files	     
