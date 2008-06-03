/*BHEADER**********************************************************************
 * (c) 1995   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.1.1.1 $
 *********************************************************************EHEADER*/

/******************************************************************************
 * SelectIJK
 *
 * AVS module to select between two read in input parflow binary files.
 *
 *****************************************************************************/

#include <stdio.h>

#include <avs.h>
#include <field.h>


/*--------------------------------------------------------------------------
 * SelectIJK
 *--------------------------------------------------------------------------*/

SelectIJK()
{
   int SelectIJK_compute();
   int p;


   AVSset_module_name("select ijk", MODULE_DATA);

   p = AVScreate_output_port("Choice Output", "choice");

   p = AVSadd_parameter("IJK Selection", "choice", "I", "I!J!K", "!");
   AVSconnect_widget(p, "radio_buttons");

   AVSset_compute_proc(SelectIJK_compute);
}

	
/*--------------------------------------------------------------------------
 * SelectIJK_compute
 *--------------------------------------------------------------------------*/

SelectIJK_compute(choice, ijk_value)
char  **choice;
char   *ijk_value;
{
    int not_fired;

    not_fired = 1;

    if ( AVSparameter_changed("IJK Selection") )
    {

       /* free old memory */
       if (*choice) free(*choice);

       switch( AVSchoice_number( "IJK Selection", ijk_value ) )
       {
          case 0 : AVSerror("SelectIJK_compute: Invalid selection.");
                   return(0);
                   break;
          case 1 : not_fired = 0;
                   *choice = malloc(sizeof(char) + 1);
                   strcpy( *choice, "I" );
                   break;
          case 2 : not_fired = 0;
                   *choice = malloc(sizeof(char) + 1);
                   strcpy( *choice, "J" );
                   break;
          case 3 : not_fired = 0;
                   *choice = malloc(sizeof(char) + 1);
                   strcpy( *choice, "K" );
                   break;
       }

   }
                
   /*-----------------------------------------------------------------------
    * clean up and return
    *-----------------------------------------------------------------------*/

   if ( not_fired )
   {
      AVSmark_output_unchanged("Choice Output");
   }
   
   return(1);
}
