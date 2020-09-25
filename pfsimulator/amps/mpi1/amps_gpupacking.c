/**********************************************************************
 *
 *  Please read the LICENSE file for the GNU Lesser General Public License.
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
 ***********************************************************************/

#include <string.h>
#include "amps.h"

//Dummy definitions if no CUDA
void amps_gpu_freebufs(){
}

char* amps_gpu_recvbuf_mpi(int pos, int size){
  (void)pos;
  (void)size;
  return NULL;
}

char* amps_gpu_recvbuf_packing(int pos, int size){
  (void)pos;
  (void)size;
  return NULL;
}

char* amps_gpu_sendbuf_mpi(int pos, int size){
  (void)pos;
  (void)size;
  return NULL;
}

char* amps_gpu_sendbuf_packing(amps_Package package){
  (void)package;
  return NULL;
}

int amps_gpupacking(amps_Invoice inv, char **buffer, int unpack){
  (void)inv;
  (void)buffer;
  (void)unpack;
  return 1;
}    
