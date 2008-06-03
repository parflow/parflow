cBHEADER***********************************************************************
c (c) 1995   The Regents of the University of California
c
c See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
c notice, contact person, and disclaimer.
c
c $Revision: 1.1.1.1 $
cEHEADER***********************************************************************

c ***************************************************************************
c *
c * Machine specific porting hacks
c *
c ***************************************************************************

#ifdef _CRAYMPP
#define dsign sign
#define d0 0
#endif
