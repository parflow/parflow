#include "amps.h"

amps_File amps_Fopen(filename, type)
char *filename;
char *type;
{
   char temp[255];

   sprintf(temp, "%s.%05d", filename, amps_Rank(amps_CommWorld));

   return fopen(temp, type);
}
