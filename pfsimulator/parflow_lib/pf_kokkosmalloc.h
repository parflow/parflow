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

/** @file
 * @brief Contains macro redefinitions for unified memory management.
 */

#ifndef PF_KOKKOSMALLOC_H
#define PF_KOKKOSMALLOC_H

#include "pf_devices.h"

/*--------------------------------------------------------------------------
 * Memory management macros for Kokkos
 *--------------------------------------------------------------------------*/

#define talloc_amps_kokkos(type, count) amps_TAlloc_managed(type, count)

#define ctalloc_amps_kokkos(type, count) amps_CTAlloc_managed(type, count)

#define tfree_amps_kokkos(ptr) amps_TFree_managed(ptr)

#define talloc_kokkos(type, count) \
        ((count) ? (type*)_talloc_device(sizeof(type) * (unsigned int)(count)) : NULL)

#define ctalloc_kokkos(type, count) \
        ((count) ? (type*)_ctalloc_device(sizeof(type) * (unsigned int)(count)) : NULL)

#define tfree_kokkos(ptr) if (ptr) _tfree_device(ptr); else {}

#define tmemcpy_kokkos(dest, src, bytes) kokkosMemCpy((char*)dest, (char*)src, (size_t)bytes);

#endif // PF_KOKKOSMALLOC_H
