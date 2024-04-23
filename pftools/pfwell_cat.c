/*BHEADER**********************************************************************
*
*  Copyright (c) 1995-2024, Lawrence Livermore National Security,
*  LLC. Produced at the Lawrence Livermore National Laboratory. Written
*  by the Parflow Team (see the CONTRIBUTORS file)
*  <parflow@lists.llnl.gov> CODE-OCEC-08-103. All rights reserved.
*
*  This file is part of Parflow. For details, see
*  http://www.llnl.gov/casc/parflow
*
*  Please read the COPYRIGHT file or Our Notice and the LICENSE file
*  for the GNU Lesser General Public License.
*
*  This program is free software; you can redistribute it and/or modify
*  it under the terms of the GNU General Public License (as published
*  by the Free Software Foundation) version 2.1 dated February 1999.
*
*  This program is distributed in the hope that it will be useful, but
*  WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms
*  and conditions of the GNU General Public License for more details.
*
*  You should have received a copy of the GNU Lesser General Public
*  License along with this program; if not, write to the Free Software
*  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
*  USA
**********************************************************************EHEADER*/

#include "pfwell_cat.h"

int main(
         int   argc,
         char *argv[])
{
  Background       *background;
  ProblemData      *problem_data;

  WellDataHeader   *well_header;
  WellDataPhysical *well_physical;
  WellDataValue    *well_value;
  WellDataStat    **well_stats, **well_stats_save;

  int              *well_action, *well_type;
  int num_wells, num_phases, num_components;

  FILE  *fd;
  int fnum, well;
  int not_eof, write_data;
  double time, current_time, offset_time;

  if (argc > 1)
  {
    background = NewBackground();
    problem_data = NewProblemData();

    if ((fd = fopen(argv[1], "r")) == NULL)
    {
      printf("Unable to open file %s\n", argv[1]);
      exit(1);
    }

    /* Read in background from target file and write out */
    ReadBackground(fd, background);
    PrintBackground(background);

    /* Read in problem data from target file and write out */
    ReadProblemData(fd, problem_data);
    PrintProblemData(problem_data);

    num_phases = ProblemDataNumPhases(problem_data);
    num_components = ProblemDataNumComponents(problem_data);
    num_wells = ProblemDataNumWells(problem_data);

    /* Get and Initialize the save data */
    well_header = NewWellDataHeader();
    well_physical = NewWellDataPhysical();
    well_value = NewWellDataValue(num_phases, num_components);

    well_stats = talloc(WellDataStat *, num_wells);
    well_stats_save = talloc(WellDataStat *, num_wells);
    for (well = 0; well < num_wells; well++)
    {
      well_stats[well] = NewWellDataStat(num_phases, num_components);
      well_stats_save[well] = NewWellDataStat(num_phases, num_components);
      InitWellDataStat(well_stats_save[well], num_phases, num_components);
    }

    well_action = talloc(int, num_wells);
    well_type = talloc(int, num_wells);

    /* Read in the well header from target file and write out */
    for (well = 0; well < num_wells; well++)
    {
      ReadWellDataHeader(fd, well_header);
      PrintWellDataHeader(well_header);

      well_action[well] = WellDataHeaderAction(well_header);
      well_type[well] = WellDataHeaderType(well_header);
    }

    not_eof = 1;
    while (not_eof)
    {
      fscanf(fd, "%lf\n", &current_time);
      if (feof(fd))
      {
        not_eof = 0;
      }
      else
      {
        time = current_time;
        printf("%f\n", time);

        for (well = 0; well < num_wells; well++)
        {
          ReadWellDataPhysical(fd, well_physical);
          PrintWellDataPhysical(well_physical);

          ReadWellDataValue(fd,
                            well_value,
                            well_action[well],
                            well_type[well],
                            num_phases,
                            num_components);

          PrintWellDataValue(well_value,
                             well_action[well],
                             well_type[well],
                             num_phases,
                             num_components);

          ReadWellDataStat(fd,
                           well_stats[well],
                           num_phases,
                           num_components);

          PrintWellDataStat(well_stats[well],
                            num_phases,
                            num_components);
        }
      }
    }

#if 0
    offset_time = time;
#endif
    offset_time = 0.0;

    for (well = 0; well < num_wells; well++)
    {
      CopyWellDataStat(well_stats_save[well],
                       well_stats[well],
                       num_phases,
                       num_components);
    }

    fclose(fd);

    fnum = 2;
    while (argc > fnum)
    {
      if ((fd = fopen(argv[fnum], "r")) == NULL)
      {
        printf("Unable to open file %s\n", argv[fnum]);
        exit(1);
      }

      /* Read in background from target file */
      ReadBackground(fd, background);

      /* Read in problem data from target file */
      ReadProblemData(fd, problem_data);

      /* Read in the well header from target file */
      for (well = 0; well < num_wells; well++)
      {
        ReadWellDataHeader(fd, well_header);
      }

      write_data = 0;
      not_eof = 1;
      while (not_eof)
      {
        fscanf(fd, "%lf\n", &current_time);
        if (feof(fd))
        {
          not_eof = 0;
        }
        else
        {
          if (write_data)
          {
            time = current_time + offset_time;
            printf("%f\n", time);
          }
          for (well = 0; well < num_wells; well++)
          {
            ReadWellDataPhysical(fd, well_physical);
            if (write_data)
            {
              PrintWellDataPhysical(well_physical);
            }

            ReadWellDataValue(fd,
                              well_value,
                              well_action[well],
                              well_type[well],
                              num_phases,
                              num_components);

            if (write_data)
            {
              PrintWellDataValue(well_value,
                                 well_action[well],
                                 well_type[well],
                                 num_phases,
                                 num_components);
            }

            ReadWellDataStat(fd,
                             well_stats[well],
                             num_phases,
                             num_components);

            if (write_data)
            {
              UpdateWellDataStat(well_stats[well],
                                 well_stats_save[well],
                                 num_phases,
                                 num_components);
              PrintWellDataStat(well_stats[well],
                                num_phases,
                                num_components);
            }
          }
          write_data = 1;
        }
      }

#if 0
      offset_time = time;
#endif
      if (write_data)
      {
        for (well = 0; well < num_wells; well++)
        {
          CopyWellDataStat(well_stats_save[well],
                           well_stats[well],
                           num_phases,
                           num_components);
        }
      }

      fclose(fd);
      fnum++;
    }

    free(well_type);
    free(well_action);

    for (well = num_wells - 1; well >= 0; well--)
    {
      FreeWellDataStat(well_stats_save[well]);
      FreeWellDataStat(well_stats[well]);
    }
    free(well_stats_save);
    free(well_stats);

    FreeWellDataValue(well_value);
    FreeWellDataPhysical(well_physical);
    FreeWellDataHeader(well_header);

    FreeProblemData(problem_data);
    FreeBackground(background);
  }
  else
  {
    printf("Usage: %s well-file_1 ... well-file_n\n", argv[0]);
  }

  return 0;
}
