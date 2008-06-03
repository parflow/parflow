#include "amps.h"

int amps_SFclose(file)
amps_File file;
{
   if(amps_Rank(amps_CommWorld))
      return 0;
   else
      return fclose(file);
}
