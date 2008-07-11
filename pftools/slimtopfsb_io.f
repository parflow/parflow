cBHEADER***********************************************************************
c (c) 1995   The Regents of the University of California
c
c See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
c notice, contact person, and disclaimer.
c
c $Revision: 1.2 $
cEHEADER***********************************************************************

      subroutine ropen(fname)
      character*(*)  fname
c
      open(30,file=fname,status='old',form='unformatted')
      return
      end

      subroutine rnindex(nindex)
      integer        nindex
c
      read(30) curtime 
      read(30) nindex
      return
      end

      subroutine rindex(index, nindex)
      integer        index(1)
      integer        nindex
c
      read(30) (index(j),j=1,nindex)

      return
      end

      subroutine rcnt(cnt, nindex)
      real           cnt(1)
      integer        nindex
c
      read(30) (cnt(j),j=1,nindex)

      return
      end

      subroutine rclose()
c
      close(30)
      return
      end



