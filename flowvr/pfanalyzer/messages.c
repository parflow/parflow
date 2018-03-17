/*BHEADER*********************************************************************
 *  This file is part of Parflow. For details, see
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
#include <string.h>

#include "messages.h"

#include <assert.h>

/**
 * Translates Variables into variable names
 */
const char *VARIABLE_TO_NAME[VARIABLE_LAST] = {
  "pressure",
  "saturation",
  "porosity",
  "manning",
  "permeability_x",
  "permeability_y",
  "permeability_z"
};

void SendActionMessage(fca_module mod, fca_port out, Action action, Variable variable,
                       void *parameter, size_t parameter_size)
{
  fca_message msg = fca_new_message(mod, sizeof(ActionMessageMetadata) + parameter_size);
  ActionMessageMetadata *amm = (ActionMessageMetadata*)fca_get_write_access(msg, 0);

  amm->action = action;
  amm->variable = variable;
  memcpy((void*)(++amm), parameter, parameter_size);

  fca_put(out, msg);
  fca_free(msg);
}

void ParseMergedMessage(fca_port port, size_t (*cb)(const void *buffer, size_t size, void *cbdata), void *cbdata)
{
  fca_message msg = fca_get(port);
  size_t s = fca_get_segment_size(msg, 0);

  if (s > 0)
  {
    const void* buffer = fca_get_read_access(msg, 0);
    void* end = buffer + s;
    while (buffer < end)
    {
      buffer += cb(buffer, s, cbdata);
//      printf("--buffer: %#010x/%#010x\n", buffer, end);
    }
    if (buffer > end)
    {
      printf("Probably a message parsing error occured!\n");
    }
  }
  fca_free(msg);
}

void SendSteerMessage(fca_module mod, fca_port out, const Action action,
                      const Variable variable,
                      int ix, int iy, int iz,
                      double *data, int nx, int ny, int nz)
{
  fca_message msg = fca_new_message(mod, sizeof(ActionMessageMetadata) +
                                    sizeof(SteerMessageMetadata) + sizeof(double) * nx * ny * nz);
  ActionMessageMetadata *amm = (ActionMessageMetadata*)fca_get_write_access(msg, 0);

  amm->action = action;
  amm->variable = variable;
  SteerMessageMetadata *m = (SteerMessageMetadata*)(amm + 1);
  m->ix = ix;
  m->iy = iy;
  m->iz = iz;
  m->nz = nz;
  m->ny = ny;
  m->nx = nx;

  // low: would be cooler if we could work directly in the message from python...
  // but: every approach would diminish the flexibility you have with python arrays.
  memcpy((void*)(m + 1), data, sizeof(double) * nx * ny * nz);

  fca_put(out, msg);
  fca_free(msg);
}


void SendLogMessage(fca_module mod, fca_port port, StampLog log[], size_t n)
{
  fca_message msg = fca_new_message(mod, 0);

  while (n--)
  {
    fca_stamp stamp = fca_get_stamp(port, log[n].stamp_name);
    fca_write_stamp(msg, stamp, (void*)&(log[n].value));
  }
  fca_put(port, msg);
  fca_free(msg);
}

Variable NameToVariable(const char *name)
{
  for (Variable res = VARIABLE_PRESSURE; res < VARIABLE_LAST; ++res)
  {
    if (strcmp(VARIABLE_TO_NAME[res], name) == 0)
      return res;
  }
  assert(0 && "Could not convert to a Variable!");
  return VARIABLE_LAST;
}

void SendEmptyMessage(fca_module mod, fca_port port)
{
  fca_message msg = fca_new_message(mod, 0);

  fca_put(port, msg);
  fca_free(msg);
}

// Autogenerate the Reader functions
GenerateMessageReaderC(Grid);
GenerateMessageReaderC(Steer);
GenerateMessageReaderC(Action);
