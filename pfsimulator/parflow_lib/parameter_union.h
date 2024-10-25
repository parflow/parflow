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
 * ParameterUnion looks for a parameter that might go by
 * different aliases or assume different types of values.
 *
 * It can be used in cases where:
 * - a physical variable may come under different names
 * - different physical quantities can provide the same
 *  information, so when one is defined, all the others
 *  are readily obtained
 * - a variable may be defined through different inputs
 *  (e.g. through a constant value or a filename).
 *
 * ParameterUnion will produce an error if none of the
 * required keys is found. At least one key/alias must be
 * defined in the input file. In case multiple aliases are
 * defined in the input file, the first key to be found is
 * returned. Key look up is done in the order in which they are
 * written into GetParameterUnion.
 *
 * Example:
 * {
 *  double arr[n];
 *  ParameterUnion par;
 *
 *  GetParameterUnion(par, {
 *    ParameterUnionDouble(0, "parameter_alias1.Value");
 *    ParameterUnionString(1, "parameter_alias1.Filename");
 *    ParameterUnionDouble(2, "parameter_alias2.Value");
 *    ParameterUnionString(3, "parameter_alias2.Filename");
 *    ParameterUnionInt(   4, "parameter_alias3.FunctionType");
 *    ParameterUnionString(5, "parameter_alias4.PredefinedFunction");
 *  });
 *
 *  for(i = 0; i < n; ++i) {
 *    switch(ParameterUnionID(par)) {
 *      case 0:
 *        // fill array with constant value
 *        arr[i] = ParameterUnionDataDouble(par);
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


#define TYPE_CHECK 1
#ifndef TYPE_CHECK

#define GetParameterUnion(par, args)                                         \
        {                                                                    \
          IDB *_database = amps_ThreadLocal(input_database);                 \
          IDB_Entry _lookup_entry;                                           \
          IDB_Entry *_result;                                                \
          char _key_list[IDB_MAX_KEY_LEN];                                   \
          ParameterUnion _tmp;                                               \
                                                                             \
          switch (1)                                                         \
          {                                                                  \
            default:                                                         \
              {                                                              \
                args                                                         \
                InputError(                                                  \
                           "Input Error: Set one of these keys:\n %s %s \n", \
                           _key_list, " ");                                  \
              }                                                              \
          }                                                                  \
          par = _tmp;                                                        \
        }


#define _ParameterUnionLookup(_key)
(                                                      \
  _lookup_entry.key = (char*)_key,                     \
  (IDB_Entry*)HBT_lookup(_database, &_lookup_entry)    \
)


#define _ParameterUnionAddKey(_key_list, _key)                 \
        {                                                      \
          strcat(_key_list, _key);                             \
          strcat(_key_list, " \n");                            \
        }


#define ParameterUnionInt(_id, _key) {                          \
          _result = _ParameterUnionLookup(_key);                \
          if (_result)                                          \
          {                                                     \
            ParameterUnionID(_tmp) = _id;                       \
            ParameterUnionDataInt(_tmp) = GetInt(_key);         \
            break;                                              \
          }                                                     \
          _ParameterUnionAddKey(_key_list, _key);               \
}


#define ParameterUnionDouble(_id, _key) {                       \
          _result = _ParameterUnionLookup(_key);                \
          if (_result)                                          \
          {                                                     \
            ParameterUnionID(_tmp) = _id;                       \
            ParameterUnionDataDouble(_tmp) = GetDouble(_key);   \
            break;                                              \
          }                                                     \
          _ParameterUnionAddKey(_key_list, _key);               \
}


#define ParameterUnionString(_id, _key) {                       \
          _result = _ParameterUnionLookup(_key);                \
          if (_result)                                          \
          {                                                     \
            ParameterUnionID(_tmp) = _id;                       \
            ParameterUnionDataString(_tmp) = GetString(_key);   \
            break;                                              \
          }                                                     \
          _ParameterUnionAddKey(_key_list, _key);               \
}


#else // TYPE_CHECK is selected

#define GetParameterUnion(par, na_types, base_str, args)               \
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


#endif