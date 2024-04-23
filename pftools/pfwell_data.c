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

#include "pfwell_data.h"

int main(
         int   argc,
         char *argv[])
{
  WellListMember   *head;
  WellListMember   *member,
    *new_member,
    *old_member = NULL;

  Background       *background;
  ProblemData      *problem_data;

  WellDataHeader   *well_header;
  WellDataPhysical *well_physical;
  WellDataValue    *well_value;
  WellDataStat     *well_stat;

  int num_wells, num_phases, num_components;
  int              *well_action, *well_type;

  FILE  *fd, **files;
  int well;
  int not_eof;
  int count;
  double time;
  char file_name[1024];

  if (argc == 3)
  {
    /*******************************************************************/
    /*                                                                 */
    /* Assemble the linked list of time and values from the input file */
    /*                                                                 */
    /*******************************************************************/
    head = NULL;

    background = NewBackground();
    problem_data = NewProblemData();

    if ((fd = fopen(argv[1], "r")) == NULL)
    {
      printf("Unable to open file %s\n", argv[1]);
      exit(1);
    }

    /* Read in background from target file and write out */
    ReadBackground(fd, background);
    printf("B");

    /* Read in problem data from target file and write out */
    ReadProblemData(fd, problem_data);
    printf("P");

    num_phases = ProblemDataNumPhases(problem_data);
    num_components = ProblemDataNumComponents(problem_data);
    num_wells = ProblemDataNumWells(problem_data);

    /* Get and Initialize the save data */
    well_header = NewWellDataHeader();
    well_physical = NewWellDataPhysical();
    well_value = NewWellDataValue(num_phases, num_components);
    well_stat = NewWellDataStat(num_phases, num_components);

    well_action = talloc(int, num_wells);
    well_type = talloc(int, num_wells);

    /* Read in the well header from target file and write out */
    for (well = 0; well < num_wells; well++)
    {
      ReadWellDataHeader(fd, well_header);

      well_action[well] = WellDataHeaderAction(well_header);
      well_type[well] = WellDataHeaderType(well_header);
      printf("h");
    }
    printf("H\n");

    count = 0;
    not_eof = 1;
    while (not_eof)
    {
      fscanf(fd, "%lf\n", &time);
      if (feof(fd))
      {
        not_eof = 0;
      }
      else
      {
        /* If there is another well record allocate a new list
         * member structure and set up the list member structure
         * with the data from the record.                        */
        member = talloc(WellListMember, 1);
        WellListMemberWellDataPhysicals(member) = talloc(WellDataPhysical *, num_wells);
        WellListMemberWellDataValues(member) = talloc(WellDataValue *, num_wells);
        WellListMemberWellDataStats(member) = talloc(WellDataStat *, num_wells);

        WellListMemberTime(member) = time;
        for (well = 0; well < num_wells; well++)
        {
          WellListMemberWellDataPhysical(member, well) = NewWellDataPhysical();
          WellListMemberWellDataValue(member, well) = NewWellDataValue(num_phases, num_components);
          WellListMemberWellDataStat(member, well) = NewWellDataStat(num_phases, num_components);

          ReadWellDataPhysical(fd,
                               well_physical);

          CopyWellDataPhysical(WellListMemberWellDataPhysical(member, well),
                               well_physical);

          ReadWellDataValue(fd,
                            well_value,
                            well_action[well],
                            well_type[well],
                            num_phases,
                            num_components);

          CopyWellDataValue(WellListMemberWellDataValue(member, well),
                            well_value,
                            well_action[well],
                            well_type[well],
                            num_phases,
                            num_components);

          ReadWellDataStat(fd,
                           well_stat,
                           num_phases,
                           num_components);


          CopyWellDataStat(WellListMemberWellDataStat(member, well),
                           well_stat,
                           num_phases,
                           num_components);
        }

        WellListMemberNextWellListMember(member) = NULL;

        /* Adjust the head or last member pointer */
        if (head == NULL)
        {
          head = member;
        }
        else
        {
          WellListMemberNextWellListMember(old_member) = member;
        }
        old_member = member;

        printf("r");
        count++;
      }
    }

    printf("R (%6d)\n", count);

    fclose(fd);

    /*******************************************************************/
    /*                                                                 */
    /*        Write out the values to output files for each well       */
    /*                                                                 */
    /*******************************************************************/
    files = talloc(FILE *, num_wells);

    for (well = 0; well < num_wells; well++)
    {
      sprintf(file_name, "%s%02d.txt", argv[2], well);
      if ((files[well] = fopen(file_name, "w")) == NULL)
      {
        printf("Unable to open file %s\n", file_name);
        exit(1);
      }
    }

    count = 0;
    member = head;
    while (member != NULL)
    {
      for (well = 0; well < num_wells; well++)
      {
        WriteWellListData(files[well],
                          WellListMemberTime(member),
                          WellListMemberWellDataStat(member, well),
                          num_phases,
                          num_components);
      }
      member = WellListMemberNextWellListMember(member);

      printf("w");
      count++;
    }
    printf("W (%6d)\n", count);

    for (well = 0; well < num_wells; well++)
    {
      fclose(files[well]);
    }

    free(files);

    /*******************************************************************/
    /*                                                                 */
    /*                      Free up the linked list                    */
    /*                                                                 */
    /*******************************************************************/
    count = 0;
    member = head;
    while (member != NULL)
    {
      new_member = WellListMemberNextWellListMember(member);

      for (well = num_wells - 1; well >= 0; well--)
      {
        FreeWellDataStat(WellListMemberWellDataStat(member, well));
        FreeWellDataValue(WellListMemberWellDataValue(member, well));
        FreeWellDataPhysical(WellListMemberWellDataPhysical(member, well));
      }
      free(WellListMemberWellDataStats(member));
      free(WellListMemberWellDataValues(member));
      free(WellListMemberWellDataPhysicals(member));

      free(member);

      member = new_member;
      printf("f");
      count++;
    }
    printf("F (%6d)\n", count);

    free(well_type);
    free(well_action);

    FreeWellDataStat(well_stat);
    FreeWellDataValue(well_value);
    FreeWellDataPhysical(well_physical);
    FreeWellDataHeader(well_header);

    FreeProblemData(problem_data);
    FreeBackground(background);
  }
  else
  {
    printf("Usage: %s well-file output-file-prefix\n", argv[0]);
  }

  return 0;
}

void WriteWellListData(
                       FILE *        fd,
                       double        time,
                       WellDataStat *well_data_stat,
                       int           num_phases,
                       int           num_components)
{
  int i, j, indx;
  double value;

  fprintf(fd, "%f", time);

  for (i = 0; i < num_phases; i++)
  {
    value = WellDataStatSaturationStat(well_data_stat, i);
    fprintf(fd, " %f", value);
  }

  for (i = 0; i < num_phases; i++)
  {
    value = WellDataStatPhaseStat(well_data_stat, i);
    fprintf(fd, " %f", value);
  }

  for (i = 0; i < num_phases; i++)
  {
    for (j = 0; j < num_components; j++)
    {
      indx = i * num_components + j;
      value = WellDataStatComponentStat(well_data_stat, indx);
      fprintf(fd, " %f", value);
    }
  }

  for (i = 0; i < num_phases; i++)
  {
    for (j = 0; j < num_components; j++)
    {
      indx = i * num_components + j;
      value = WellDataStatConcentrationStat(well_data_stat, indx);
      fprintf(fd, " %f", value);
    }
  }

  fprintf(fd, "\n");
}
