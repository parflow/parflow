#include "amps.h"

amps_File amps_SFopen(filename, type)
char *filename;
char *type;
{

   if(amps_Rank(amps_CommWorld))
      return (amps_File)1;
   else
      return fopen(filename, type);
}
