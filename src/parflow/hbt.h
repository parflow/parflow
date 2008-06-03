/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/
#ifndef _hbt_HEADER
#define _hbt_HEADER

/*****************************************************************************/
/* Height Balanced Tree Information                                          */
/*****************************************************************************/

/*===========================================================================*/
/* These are the return codes.                                               */
/*===========================================================================*/
#define HBT_NOT_FOUND 0
#define HBT_FOUND 1
#define HBT_INSERTED 2
#define HBT_DELETED 3
#define HBT_MEMORY_ERROR 4

#define TRUE 1
#define FALSE 0

/*===========================================================================*/
/* This is an actual node on the HBT.                                    */
/*===========================================================================*/
typedef struct _HBT_element
{
  void *obj;

  short balance;

  struct _HBT_element *left;
  struct _HBT_element *right;

} HBT_element;

/*===========================================================================*/
/* This is the "head" structure for HBT's.                               */
/*===========================================================================*/
typedef struct _HBT
{

  unsigned int height;
  unsigned int num;
  
  HBT_element *root;


  int (*compare)();
  void (*free)();
  void (*printf)();
  int  (*scanf)();

  int malloc_flag;
  
} HBT;

#endif

