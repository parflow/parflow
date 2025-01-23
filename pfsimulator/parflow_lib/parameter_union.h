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

/*
 * ParameterUnion is an attempt to prevent repeated code
 * when reading information from the input script.
 * It should be used in cases where the values for a
 * variable can be set from different inputs. For example,
 * a variable could be set with a constant value, read
 * from a file, or given by a predefined function.
 *
 * The macro arguments are:
 *
 * @param par        ParameterUnion structure that stores
 *                   the values read from the input script.
 * @param base_str   Base of the key string that is looked
 *                   for in the database. Keys are created
 *                   by adding suffixes to the base.
 * @param na_types   Expected values that the input key
 *                   "`base_str`.Type" can assume. This
 *                   selects which of the subsequent
 *                   suffixes will be read in the input script.
 * @param args       Body macros that tell what kind of
 *                   variable is expected to be read, what
 *                   key value they correspond to in the
 *                   `na_types`
 *
 *
 * Example:
 * {
 *  ParameterUnion par;
 *  NameArray na_types = NA_NewNameArray("Constant
 *                                        PFBFile
 *                                        PredefinedFunction
 *                                        Option");
 *
 *  GetParameterUnion(par, "base.string", na_types,
 *    ParameterUnionDouble(0, "Value")
 *    ParameterUnionString(1, "Filename")
 *    ParameterUnionString(2, "PredefinedFunction")
 *    ParameterUnionInt(3, "Option")
 *  );
 *
 *  for(i = 0; i < n; ++i) {
 *    switch(ParameterUnionID(par)) {
 *      case 0:
 *        double value = ParameterUnionDataDouble(par);
 *      case 1:
 *        // fill array with values from a file
 *        char *filename = ParameterUnionDataString(par);
 *      (...)
 *    }
 *  }
 * }
 *
 */

#include <math.h>

typedef union {
  int intg;
  double dobl;
  char  *strg;
} _ParameterUnionData;


typedef struct {
  _ParameterUnionData data; // parameter data
  int id;                  // identifies which key was found
} ParameterUnion;


#define ParameterUnionID(par)         (par.id)
#define ParameterUnionData(par)       (par.data)
#define ParameterUnionDataInt(par)    (par.data.intg)
#define ParameterUnionDataDouble(par) (par.data.dobl)
#define ParameterUnionDataString(par) (par.data.strg)


#define GetParameterUnion(par, base_str, na_types, args)               \
        {                                                              \
          char _base_str[IDB_MAX_KEY_LEN];                             \
          char _full_key[IDB_MAX_KEY_LEN];                             \
          char *_type_str;                                             \
          ParameterUnion _tmp;                                         \
                                                                       \
          strcpy(_base_str, base_str);                                 \
          sprintf(_full_key, _base_str, "Type");                       \
          _type_str = GetString(_full_key);                            \
                                                                       \
          int _type =                                                  \
            NA_NameToIndexExitOnError(na_types, _type_str, _full_key); \
                                                                       \
          switch (_type)                                               \
          {                                                            \
          args                                                         \
            default:                                                   \
              {                                                        \
                ParameterUnionDataInt(_tmp) = 0;                       \
                break;                                                 \
              }                                                        \
          }                                                            \
          par = _tmp;                                                  \
        }


#define ParameterUnionInt(_id, _key)                                 \
          case _id:                                                  \
            {                                                        \
              sprintf(_full_key, _base_str, _key);                   \
              ParameterUnionID(_tmp) = _id;                          \
              ParameterUnionDataInt(_tmp) = GetInt(_full_key);       \
              break;                                                 \
            }


#define ParameterUnionDouble(_id, _key)                              \
          case _id:                                                  \
            {                                                        \
              sprintf(_full_key, _base_str, _key);                   \
              ParameterUnionID(_tmp) = _id;                          \
              ParameterUnionDataDouble(_tmp) = GetDouble(_full_key); \
              break;                                                 \
            }


#define ParameterUnionString(_id, _key)                              \
          case _id:                                                  \
            {                                                        \
              sprintf(_full_key, _base_str, _key);                   \
              ParameterUnionID(_tmp) = _id;                          \
              ParameterUnionDataString(_tmp) = GetString(_full_key); \
              break;                                                 \
            }
