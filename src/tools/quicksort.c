/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1 $
 *********************************************************************EHEADER*/

/*--------------------------------------------------------------------------
 * The following type must be defined to compile QuickSort:
 *
 *   EltType
 *
 *
 * The following macros (or functions) must be defined to compile QuickSort:
 *
 *   QSORT_CROSSOVER
 *     This is an integer value (>= 3) which determines when the
 *     quicksort algorithm "crosses over" to an insertion sort
 *
 *   CompareLessThan(result, left_operand, right_operand)
 *     Returns the result of (left_operand < right_operand)
 *
 *   CompareGreaterThan(result, left_operand, right_operand)
 *     Returns the result of (left_operand > right_operand)
 *
 *   Swap(array, i, j, tmp)
 *     Swaps array[i] and array[j].  The argument `tmp' is used as
 *     temporary space in the swap and must be of the same type as
 *     both array[i] and array[j].
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * QuickSort:
 *   This routine is a recursive routine for quickly sorting `array'.
 *   If the additional argument called `permute' is not NULL, it is
 *   sorted identically to `array'.  This integer array is usually used
 *   to keep track of the permuation of elements in `array' resulting
 *   from the sorting.
 *--------------------------------------------------------------------------*/

void      QuickSort(first, last, array, permute)
int       first;
int       last;
EltType  *array;
int      *permute;
{
   EltType  tmp_vert;
   int      tmp_index;

   int      i, j, compare_result;


   /*-----------------------------------------------------------------------
    * For large lists, use quicksort
    *-----------------------------------------------------------------------*/

   if ((last - first) > QSORT_CROSSOVER)
   {
      /*--------------------------------------------------------------------
       * Partition the elements
       *--------------------------------------------------------------------*/

      /* set element `first' (the partition element) to a "median" element */
      i = (first + last) / 2;
      j = (first + 1);
      Swap(array, i, j, tmp_vert);
      if (permute)
	 Swap(permute, i, j, tmp_index);
      CompareGreaterThan(compare_result, array[j], array[last]);
      if (compare_result)
      {
	 Swap(array, j, last, tmp_vert);
	 if (permute)
	    Swap(permute, j, last, tmp_index);
      }
      CompareGreaterThan(compare_result, array[first], array[last]);
      if (compare_result)
      {
	 Swap(array, first, last, tmp_vert);
	 if (permute)
	    Swap(permute, first, last, tmp_index);
      }
      CompareGreaterThan(compare_result, array[j], array[first]);
      if (compare_result)
      {
	 Swap(array, j, first, tmp_vert);
	 if (permute)
	    Swap(permute, j, first, tmp_index);
      }

      i = first + 1;
      j = last;
      while (1)
      {
	 /* find element `i' that is >= element `first' */
	 while (!(i > last))
	 {
	    CompareLessThan(compare_result, array[i], array[first]);
	    if (!compare_result)
	       break;
	    i++;
	 }

	 /* find element `j' that is <= element `first' */
	 while (1)
	 {
	    CompareGreaterThan(compare_result, array[j], array[first]);
	    if (!compare_result)
	       break;
	    j--;
	 }

	 /* if (i < j), swap values; else we are done */
	 if (i < j)
	 {
	    Swap(array, i, j, tmp_vert);
	    if (permute)
	       Swap(permute, i, j, tmp_index);
	    i++;
	    j--;
	 }
	 else
	    break;
      }

      /* put element `first' in its place */
      Swap(array, first, j, tmp_vert);
      if (permute)
	 Swap(permute, first, j, tmp_index);

      /*--------------------------------------------------------------------
       * Sort the sublists
       *--------------------------------------------------------------------*/

      QuickSort(first, (j-1), array, permute);
      QuickSort((j+1),  last, array, permute);
   }

   /*-----------------------------------------------------------------------
    * For small lists, use "insertion sort" for efficiency
    *-----------------------------------------------------------------------*/

   else
   {
      for (j = (first+1); j <= last; j++)
      {
	 i = j - 1;
	 tmp_vert  = array[j];
	 if (permute)
	    tmp_index = permute[j];
	 while (i >= first)
	 {
	    CompareLessThan(compare_result, tmp_vert, array[i]);
	    if (!compare_result)
	       break;

	    array[i+1]   = array[i];
	    if (permute)
	       permute[i+1] = permute[i];

	    i--;
	 }

	 array[i+1]   = tmp_vert;
	 if (permute)
	    permute[i+1] = tmp_index;
      }
   }
}

