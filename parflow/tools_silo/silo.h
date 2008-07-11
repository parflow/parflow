/*

                           Copyright 1991 - 2002
                The Regents of the University of California.
                            All rights reserved.

This work was produced at the University of California, Lawrence
Livermore National Laboratory (UC LLNL) under contract no.  W-7405-ENG-48
(Contract 48) between the U.S. Department of Energy (DOE) and The Regents
of the University of California (University) for the operation of UC LLNL.
Copyright is reserved to the University for purposes of controlled
dissemination, commercialization through formal licensing, or other
disposition under terms of Contract 48; DOE policies, regulations and
orders; and U.S. statutes.  The rights of the Federal Government are
reserved under Contract 48 subject to the restrictions agreed upon by
DOE and University.

                                DISCLAIMER

This software was prepared as an account of work sponsored by an agency
of the United States Government. Neither the United States Government
nor the University of California nor any of their employees, makes any
warranty, express or implied, or assumes any liability or responsiblity
for the accuracy, completeness, or usefullness of any information,
apparatus, product, or process disclosed, or represents that its use
would not infringe privately owned rights. Reference herein to any
specific commercial products, process, or service by trade name, trademark,
manufacturer, or otherwise, does not necessarily constitute or imply its
endorsement, recommendation, or favoring by the United States Government
or the University of California. The views and opinions of authors
expressed herein do not necessarily state or reflect those of the United
States Government or the University of California, and shall not be used
for advertising or product endorsement purposes.

*/

/*
 * SILO Public header file.
 *
 * This header file defines public constants and public prototypes.
 * Before including this file, the application should define
 * which file formats will be used.
 *
 * WARNING: The `#define' statements in this file are used when
 *      generating the Fortran include file `silo.inc'.  Any
 *     such symbol that should not be an integer parameter
 *     in the Fortran include file should have the symbol
 *     `NO_FORTRAN_DEFINE' on the same line.  #define statements
 *     that define macros (or any value not beginning with
 *     one of [a-zA-Z0-9_]) are ignored.
 */
#ifndef SILO_H
#define SILO_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>

/*-------------------------------------------------------------------------
 * Drivers.  This is a list of every driver that a user could use.  Not all of
 * them are necessarily compiled into the library.  However, users are free
 * to try without getting compilation errors.  They are listed here so that
 * silo.h doesn't have to be generated every time the library is recompiled.
 *--------------------------------------------------------------------------*/
#define DB_NETCDF 0
#define DB_PDB 2
#define DB_TAURUS 3
#define DB_SDX 4
#define DB_UNKNOWN 5
#define DB_DEBUG 6
#define DB_HDF5 7
#define DB_EXODUS 9

#define NO_FORTRAN_DEFINE       /*mkinc ignores these lines. */

/*-------------------------------------------------------------------------
 * Other library-wide constants.
 *-------------------------------------------------------------------------*/
#define DB_NFILES       256         /*Max simultaneously open files */
#define DB_NFILTERS     32          /*Number of filters defined */

/*-------------------------------------------------------------------------
 * Constants.  All of these constants are always defined in the application.
 * Each group of constants defined here are small integers used as an index
 * into an array.  Many of the groups have a total count of items in the
 * group that will be used for array allocation and error checking--don't
 * forget to increment the value when adding a new item to a constant group.
 *-------------------------------------------------------------------------
 */

/* The following identifiers are for use with the DBDataReadMask() call.  They
 * specify what portions of the data beyond the metadata is allocated
 * and read.  */
#define DBAll                0xffffffff
#define DBNone               0x00000000
#define DBCalc               0x00000001
#define DBMatMatnos          0x00000002
#define DBMatMatlist         0x00000004
#define DBMatMixList         0x00000008
#define DBCurveArrays        0x00000010
#define DBPMCoords           0x00000020
#define DBPVData             0x00000040
#define DBQMCoords           0x00000080
#define DBQVData             0x00000100
#define DBUMCoords           0x00000200
#define DBUMFacelist         0x00000400
#define DBUMZonelist         0x00000800
#define DBUVData             0x00001000
#define DBFacelistInfo       0x00002000
#define DBZonelistInfo       0x00004000
#define DBMatMatnames        0x00008000
#define DBUMGlobNodeNo       0x00010000
#define DBZonelistGlobZoneNo 0x00020000

/* Macros used for exporting symbols on Win32 systems. */
#ifndef SILO_API
#ifdef _WIN32
/* Make Silo a DLL by default. */
#ifdef SILO_STATIC_LIBRARY
#define SILO_API
#else
#ifdef SILO_EXPORTS
#define SILO_API __declspec(dllexport)
#else
#define SILO_API __declspec(dllimport)
#endif
#endif
#else
#define SILO_API
#endif
#endif

/* Objects that can be stored in a data file */
typedef enum {
    DB_INVALID_OBJECT= -1,       /*causes enum to be signed, do not remove,
                                   space before minus sign necessary for lint*/
    DB_QUADMESH=500,
    DB_QUADVAR=501,
    DB_UCDMESH=510,
    DB_UCDVAR=511,
    DB_MULTIMESH=520,
    DB_MULTIVAR=521,
    DB_MULTIMAT=522,
    DB_MULTIMATSPECIES=523,
    DB_MULTIBLOCKMESH=DB_MULTIMESH,
    DB_MULTIBLOCKVAR=DB_MULTIVAR,
    DB_MATERIAL=530,
    DB_MATSPECIES=531,
    DB_FACELIST=550,
    DB_ZONELIST=551,
    DB_EDGELIST=552,
    DB_PHZONELIST=553,
    DB_CURVE=560,
    DB_POINTMESH=570,
    DB_POINTVAR=571,
    DB_ARRAY=580,
    DB_DIR=600,
    DB_VARIABLE=610,
    DB_USERDEF=700
} DBObjectType;

/* Data types */
typedef enum {
    DB_INT=16,
    DB_SHORT=17,
    DB_LONG=18,
    DB_FLOAT=19,
    DB_DOUBLE=20,
    DB_CHAR=21,
    DB_NOTYPE=25           /*unknown type */
} DBdatatype;

/* Flags for DBCreate */
#define         DB_CLOBBER      0
#define         DB_NOCLOBBER    1

/* Flags for DBOpen */
#define         DB_READ         1
#define         DB_APPEND       2

/* Target machine for DBCreate */
#define         DB_LOCAL        0
#define         DB_SUN3         10
#define         DB_SUN4         11
#define         DB_SGI          12
#define         DB_RS6000       13
#define         DB_CRAY         14
#define         DB_INTEL        15

/* Options */
#define DBOPT_ALIGN             260
#define DBOPT_COORDSYS          262
#define DBOPT_CYCLE             263
#define DBOPT_FACETYPE          264
#define DBOPT_HI_OFFSET         265
#define DBOPT_LO_OFFSET         266
#define DBOPT_LABEL             267
#define DBOPT_XLABEL            268
#define DBOPT_YLABEL            269
#define DBOPT_ZLABEL            270
#define DBOPT_MAJORORDER        271
#define DBOPT_NSPACE            272
#define DBOPT_ORIGIN            273
#define DBOPT_PLANAR            274
#define DBOPT_TIME              275
#define DBOPT_UNITS             276
#define DBOPT_XUNITS            277
#define DBOPT_YUNITS            278
#define DBOPT_ZUNITS            279
#define DBOPT_DTIME             280
#define DBOPT_USESPECMF         281
#define DBOPT_XVARNAME          282
#define DBOPT_YVARNAME          283
#define DBOPT_ZVARNAME          284
#define DBOPT_ASCII_LABEL       285
#define DBOPT_MATNOS            286
#define DBOPT_NMATNOS           287
#define DBOPT_MATNAME           288
#define DBOPT_NMAT              289
#define DBOPT_NMATSPEC          290
#define DBOPT_BASEINDEX         291 /* quad meshes for node and zone */
#define DBOPT_ZONENUM           292 /* ucd meshes for zone */
#define DBOPT_NODENUM           293 /* ucd meshes for node */
#define DBOPT_BLOCKORIGIN       294
#define DBOPT_GROUPNUM          295
#define DBOPT_GROUPORIGIN       296
#define DBOPT_NGROUPS           297
#define DBOPT_MATNAMES          298
#define DBOPT_EXTENTS_SIZE      299
#define DBOPT_EXTENTS           300
#define DBOPT_MATCOUNTS         301
#define DBOPT_MATLISTS          302
#define DBOPT_MIXLENS           303
#define DBOPT_ZONECOUNTS        304
#define DBOPT_HAS_EXTERNAL_ZONES 305
#define DBOPT_PHZONELIST        306

/* Error trapping method */
#define         DB_TOP          0 /*default--API traps  */
#define         DB_NONE         1 /*no errors trapped  */
#define         DB_ALL          2 /*all levels trap (traceback) */
#define         DB_ABORT        3 /*abort() is called  */
#define         DB_SUSPEND      4 /*suspend error reporting temporarily */
#define         DB_RESUME       5 /*resume normal error reporting */

/* Errors */
#define     E_NOERROR   0       /*No error   */
#define     E_BADFTYPE  1       /*Bad file type   */
#define     E_NOTIMP    2       /*Callback not implemented */
#define     E_NOFILE    3       /*No data file specified    */
#define     E_INTERNAL  5       /*Internal error        */
#define     E_NOMEM     6       /*Not enough memory     */
#define     E_BADARGS   7       /*Bad argument to function  */
#define     E_CALLFAIL  8       /*Low-level function failure    */
#define     E_NOTFOUND  9       /*Object not found      */
#define     E_TAURSTATE 10      /*Taurus: database state error  */
#define     E_MSERVER   11      /*SDX: too many connections */
#define     E_PROTO     12      /*SDX: protocol error       */
#define     E_NOTDIR    13      /*Not a directory       */
#define     E_MAXOPEN   14      /*Too many open files  */
#define     E_NOTFILTER 15      /*Filter(s) not found  */
#define     E_MAXFILTERS    16  /*Too many filters  */
#define     E_FEXIST    17      /*File already exists  */
#define     E_FILEISDIR 18      /*File is actually a directory */
#define     E_FILENOREAD    19  /*File lacks read permission. */
#define     E_SYSTEMERR 20      /*System level error occured. */
#define     E_FILENOWRITE 21    /*File lacks write permission. */
#define     E_INVALIDNAME 22     /* Variable name is invalid */
#define     E_NERRORS   50

/* Definitions for MAJOR_ORDER */
#define  DB_ROWMAJOR            0
#define  DB_COLMAJOR            1

/* Definitions for COORD_TYPE */
#define  DB_COLLINEAR           130
#define  DB_NONCOLLINEAR        131
#define  DB_QUAD_RECT           DB_COLLINEAR
#define  DB_QUAD_CURV           DB_NONCOLLINEAR

/* Definitions for CENTERING */
#define  DB_NOTCENT             0
#define  DB_NODECENT            110
#define  DB_ZONECENT            111
#define  DB_FACECENT            112

/* Definitions for COORD_SYSTEM */
#define  DB_CARTESIAN           120
#define  DB_CYLINDRICAL         121
#define  DB_SPHERICAL           122
#define  DB_NUMERICAL           123
#define  DB_OTHER               124

/* Definitions for ZONE FACE_TYPE */
#define  DB_RECTILINEAR         100
#define  DB_CURVILINEAR         101

/* Definitions for PLANAR */
#define  DB_AREA                140
#define  DB_VOLUME              141

/* Definitions for flag values */
#define DB_ON                    1000
#define DB_OFF                  -1000

/* Miscellaneous constants */
#define     DB_F77NULL  (-99)   /*Fortran NULL pointer      */
#define     DB_F77NULLSTRING  "NULLSTRING"  /* FORTRAN STRING */

/*-------------------------------------------------------------------------
 * Index selection macros
 *-------------------------------------------------------------------------
 */
#define I4D(s,i,j,k,l) (l)*s[3]+(k)*s[2]+(j)*s[1]+(i)*s[0]
#define I3D(s,i,j,k)   (k)*s[2]+(j)*s[1]+(i)*s[0]
#define I2D(s,i,j)     (j)*s[1]+(i)*s[0]

/*-------------------------------------------------------------------------
 * Structures (just the public parts).
 *-------------------------------------------------------------------------
 */

/*
 * Database table of contents for the current directory only.
 */
typedef struct {

    char         **curve_names;
    int            ncurve;

    char         **multimesh_names;
    int            nmultimesh;

    char         **multivar_names;
    int            nmultivar;

    char         **multimat_names;
    int            nmultimat;

    char         **multimatspecies_names;
    int            nmultimatspecies;

    char         **qmesh_names;
    int            nqmesh;

    char         **qvar_names;
    int            nqvar;

    char         **ucdmesh_names;
    int            nucdmesh;

    char         **ucdvar_names;
    int            nucdvar;

    char         **ptmesh_names;
    int            nptmesh;

    char         **ptvar_names;
    int            nptvar;

    char         **mat_names;
    int            nmat;

    char         **matspecies_names;
    int            nmatspecies;

    char         **var_names;
    int            nvar;

    char         **obj_names;
    int            nobj;

    char         **dir_names;
    int            ndir;

    char         **array_names;
    int            narrays;

} DBtoc;

/*----------------------------------------------------------------------------
 * Database Curve Object
 *--------------------------------------------------------------------------
 */
typedef struct {
/*----------- X vs. Y (Curve) Data -----------*/
    int            id;          /* Identifier for this object */
    int            datatype;    /* Datatype for x and y (float, double) */
    int            origin;      /* '0' or '1' */
    char          *title;       /* Title for curve */
    char          *xvarname;    /* Name of domain (x) variable */
    char          *yvarname;    /* Name of range  (y) variable */
    char          *xlabel;      /* Label for x-axis */
    char          *ylabel;      /* Label for y-axis */
    char          *xunits;      /* Units for domain */
    char          *yunits;      /* Units for range  */
    float         *x;           /* Domain values for curve */
    float         *y;           /* Range  values for curve */
    int            npts;        /* Number of points in curve */
} DBcurve;

typedef struct {
/*----------- Point Mesh -----------*/
    int            id;          /* Identifier for this object */
    int            block_no;    /* Block number for this mesh */
    int            group_no;    /* Block group number for this mesh */
    char          *name;        /* Name associated with this mesh */
    int            cycle;       /* Problem cycle number */
    char          *units[3];    /* Units for each axis */
    char          *labels[3];   /* Labels for each axis */
    char          *title;       /* Title for curve */

    float         *coords[3];   /* Coordinate values */
    float          time;        /* Problem time */
    double         dtime;       /* Problem time, double data type */
   /*
    * The following two fields really only contain 3 elements.  However, silo
    * contains a bug in PJ_ReadVariable() as called by DBGetPointmesh() which
    * can cause three doubles to be stored there instead of three floats.
    */
    float          min_extents[6];  /* Min mesh extents [ndims] */
    float          max_extents[6];  /* Max mesh extents [ndims] */

    int            datatype;    /* Datatype for coords (float, double) */
    int            ndims;       /* Number of computational dimensions */
    int            nels;        /* Number of elements in mesh */
    int            origin;      /* '0' or '1' */
} DBpointmesh;

/*----------------------------------------------------------------------------
 * Multi-Block Mesh Object
 *--------------------------------------------------------------------------
 */
typedef struct {
/*----------- Multi-Block Mesh -----------*/
    int            id;          /* Identifier for this object */
    int            nblocks;     /* Number of blocks in mesh */
    int            ngroups;     /* Number of block groups in mesh */
    int           *meshids;     /* Array of mesh-ids which comprise mesh */
    char         **meshnames;   /* Array of mesh-names for meshids */
    int           *meshtypes;   /* Array of mesh-type indicators [nblocks] */
    int           *dirids;      /* Array of directory ID's which contain blk */
    int            blockorigin; /* Origin (0 or 1) of block numbers */
    int            grouporigin; /* Origin (0 or 1) of group numbers */
    int            extentssize; /* size of each extent tuple */
    double        *extents;     /* min/max extents of coords of each block */
    int           *zonecounts;  /* array of zone counts for each block */
    int           *has_external_zones;  /* external flags for each block */
} DBmultimesh;

/*----------------------------------------------------------------------------
 * Multi-Block Variable Object
 *--------------------------------------------------------------------------
 */
typedef struct {
/*----------- Multi-Block Variable -----------*/
    int            id;          /* Identifier for this object  */
    int            nvars;       /* Number of variables   */
    int            ngroups;     /* Number of block groups in mesh */
    char         **varnames;    /* Variable names   */
    int           *vartypes;    /* variable types   */
    int            blockorigin; /* Origin (0 or 1) of block numbers */
    int            grouporigin; /* Origin (0 or 1) of group numbers */
    int            extentssize; /* size of each extent tuple */
    double        *extents;     /* min/max extents of each block */
} DBmultivar;

/*-------------------------------------------------------------------------
 * Multi-material
 *-------------------------------------------------------------------------
 */
typedef struct {
    int            id;          /* Identifier for this object  */
    int            nmats;       /* Number of materials   */
    int            ngroups;     /* Number of block groups in mesh */
    char         **matnames;    /* Material names   */
    int            blockorigin; /* Origin (0 or 1) of block numbers */
    int            grouporigin; /* Origin (0 or 1) of group numbers */
    int           *mixlens;     /* array of mixlen values in each mat */
    int           *matcounts;   /* counts of unique materials in each block */
    int           *matlists;    /* list of materials in each block */
} DBmultimat;

/*-------------------------------------------------------------------------
 * Multi-species
 *-------------------------------------------------------------------------
 */
typedef struct {
    int            id;          /* Identifier for this object  */
    int            nspec;       /* Number of species   */
    int            ngroups;     /* Number of block groups in mesh */
    char         **specnames;   /* Species names   */    
    int            blockorigin; /* Origin (0 or 1) of block numbers */
    int            grouporigin; /* Origin (0 or 1) of group numbers */
} DBmultimatspecies;

/*----------------------------------------------------------------------
 *  Definitions for the FaceList, ZoneList, and EdgeList structures
 *  used for describing UCD meshes.
 *----------------------------------------------------------------------
 */

#define DB_ZONETYPE_BEAM        10

#define DB_ZONETYPE_POLYGON     20
#define DB_ZONETYPE_TRIANGLE    23
#define DB_ZONETYPE_QUAD        24

#define DB_ZONETYPE_POLYHEDRON  30
#define DB_ZONETYPE_TET         34
#define DB_ZONETYPE_PYRAMID     35
#define DB_ZONETYPE_PRISM       36
#define DB_ZONETYPE_HEX         38

typedef struct {
    int            ndims;       /* Number of dimensions (2,3) */
    int            nzones;      /* Number of zones in list */
    int            nshapes;     /* Number of zone shapes */
    int           *shapecnt;    /* [nshapes] occurences of each shape */
    int           *shapesize;   /* [nshapes] Number of nodes per shape */
    int           *shapetype;   /* [nshapes] Type of shape */
    int           *nodelist;    /* Sequent lst of nodes which comprise zones */
    int            lnodelist;   /* Number of nodes in nodelist */
    int            origin;      /* '0' or '1' */
    int            min_index;   /* Index of first real zone */
    int            max_index;   /* Index of last real zone */

/*--------- Optional zone attributes ---------*/
    int           *zoneno;      /* [nzones] zone number of each zone */
    int           *gzoneno;     /* [nzones] global zone number of each zone */
} DBzonelist;

typedef struct {
    int            nfaces;      /* Number of faces in facelist (aka "facetable") */
    int           *nodecnt;     /* Count of nodes in each face */
    int            lnodelist;   /* Length of nodelist used to construct faces */
    int           *nodelist;    /* List of nodes used in all faces */
    char          *extface;     /* boolean flag indicating if a face is external */
    int            nzones;      /* Number of zones in this zonelist */
    int           *facecnt;     /* Count of faces in each zone */
    int            lfacelist;   /* Length of facelist used to construct zones */
    int           *facelist;    /* List of faces used in all zones */
    int            origin;      /* '0' or '1' */
    int            lo_offset;   /* Index of first non-ghost zone */
    int            hi_offset;   /* Index of last non-ghost zone */

/*--------- Optional zone attributes ---------*/
    int           *zoneno;      /* [nzones] zone number of each zone */
    int           *gzoneno;     /* [nzones] global zone number of each zone */
} DBphzonelist;

typedef struct {
/*----------- Required components ------------*/
    int            ndims;       /* Number of dimensions (2,3) */
    int            nfaces;      /* Number of faces in list */
    int            origin;      /* '0' or '1' */
    int           *nodelist;    /* Sequent list of nodes comprise faces */
    int            lnodelist;   /* Number of nodes in nodelist */

/*----------- 3D components ------------------*/
    int            nshapes;     /* Number of face shapes */
    int           *shapecnt;    /* [nshapes] Num of occurences of each shape */
    int           *shapesize;   /* [nshapes] Number of nodes per shape */

/*----------- Optional type component---------*/
    int            ntypes;      /* Number of face types */
    int           *typelist;    /* [ntypes] Type ID for each type */
    int           *types;       /* [nfaces] Type info for each face */

/*--------- Optional node attributes ---------*/
    int           *nodeno;      /* [lnodelist] node number of each node */

/*----------- Optional zone-reference component---------*/
    int           *zoneno;      /* [nfaces] Zone number for each face */
} DBfacelist;

typedef struct {
    int            ndims;       /* Number of dimensions (2,3) */
    int            nedges;      /* Number of edges */
    int           *edge_beg;    /* [nedges] */
    int           *edge_end;    /* [nedges] */
    int            origin;      /* '0' or '1' */
} DBedgelist;

typedef struct {
/*----------- Quad Mesh -----------*/
    int            id;          /* Identifier for this object */
    int            block_no;    /* Block number for this mesh */
    int            group_no;    /* Block group number for this mesh */
    char          *name;        /* Name associated with mesh */
    int            cycle;       /* Problem cycle number */
    int            coord_sys;   /* Cartesian, cylindrical, spherical */
    int            major_order; /* 1 indicates row-major for multi-d arrays */
    int            stride[3];   /* Offsets to adjacent elements  */
    int            coordtype;   /* Coord array type: collinear,
                                 * non-collinear */
    int            facetype;    /* Zone face type: rect, curv */
    int            planar;      /* Sentinel: zones represent area or volume? */

    float         *coords[3];   /* Mesh node coordinate ptrs [ndims] */
    int            datatype;    /* Type of coordinate arrays (double,float) */
    float          time;        /* Problem time */
    double         dtime;       /* Problem time, double data type */
   /*
    * The following two fields really only contain 3 elements.  However, silo
    * contains a bug in PJ_ReadVariable() as called by DBGetQuadmesh() which
    * can cause three doubles to be stored there instead of three floats.
    */
    float          min_extents[6];  /* Min mesh extents [ndims] */
    float          max_extents[6];  /* Max mesh extents [ndims] */

    char          *labels[3];   /* Label associated with each dimension */
    char          *units[3];    /* Units for variable, e.g, 'mm/ms' */
    int            ndims;       /* Number of computational dimensions */
    int            nspace;      /* Number of physical dimensions */
    int            nnodes;      /* Total number of nodes */

    int            dims[3];     /* Number of nodes per dimension */
    int            origin;      /* '0' or '1' */
    int            min_index[3];   /* Index in each dimension of 1st
                                    * non-phoney */
    int            max_index[3];   /* Index in each dimension of last
                                    * non-phoney */
    int            base_index[3];  /* Lowest real i,j,k value for this block */
    int            start_index[3]; /* i,j,k values corresponding to original
                                    * mesh */
    int            size_index[3];  /* Number of nodes per dimension for 
                                    * original mesh */
} DBquadmesh;

typedef struct {
/*----------- Unstructured Cell Data (UCD) Mesh -----------*/
    int            id;          /* Identifier for this object */
    int            block_no;    /* Block number for this mesh */
    int            group_no;    /* Block group number for this mesh */
    char          *name;        /* Name associated with mesh */
    int            cycle;       /* Problem cycle number */
    int            coord_sys;   /* Coordinate system */
    char          *units[3];    /* Units for variable, e.g, 'mm/ms' */
    char          *labels[3];   /* Label associated with each dimension */

    float         *coords[3];   /* Mesh node coordinates */
    int            datatype;    /* Type of coordinate arrays (double,float) */
    float          time;        /* Problem time */
    double         dtime;       /* Problem time, double data type */
   /*
    * The following two fields really only contain 3 elements.  However, silo
    * contains a bug in PJ_ReadVariable() as called by DBGetUcdmesh() which
    * can cause three doubles to be stored there instead of three floats.
    */
    float          min_extents[6];  /* Min mesh extents [ndims] */
    float          max_extents[6];  /* Max mesh extents [ndims] */

    int            ndims;       /* Number of computational dimensions */
    int            nnodes;      /* Total number of nodes */
    int            origin;      /* '0' or '1' */

    DBfacelist    *faces;       /* Data structure describing mesh faces */
    DBzonelist    *zones;       /* Data structure describing mesh zones */
    DBedgelist    *edges;       /* Data struct describing mesh edges
                                 * (option) */

/*--------- Optional node attributes ---------*/
    int           *gnodeno;     /* [nnodes] global node number of each node */

/*--------- Optional zone attributes ---------*/
    int           *nodeno;      /* [nnodes] node number of each node */

/*--------- Optional polyhedral zonelist ---------*/
    DBphzonelist  *phzones;     /* Data structure describing mesh zones */

} DBucdmesh;

/*----------------------------------------------------------------------------
 * Database Mesh-Variable Object
 *---------------------------------------------------------------------------
 */
typedef struct {
/*----------- Quad Variable -----------*/
    int            id;          /* Identifier for this object */
    char          *name;        /* Name of variable */
    char          *units;       /* Units for variable, e.g, 'mm/ms' */
    char          *label;       /* Label (perhaps for editing purposes) */
    int            cycle;       /* Problem cycle number */
    int            meshid;      /* Identifier for associated mesh */

    float        **vals;        /* Array of pointers to data arrays */
    int            datatype;    /* Type of data pointed to by 'val' */
    int            nels;        /* Number of elements in each array */
    int            nvals;       /* Number of arrays pointed to by 'vals' */
    int            ndims;       /* Rank of variable */
    int            dims[3];     /* Number of elements in each dimension */

    int            major_order; /* 1 indicates row-major for multi-d arrays */
    int            stride[3];   /* Offsets to adjacent elements  */
    int            min_index[3];  /* Index in each dimension of 1st
                                   * non-phoney */
    int            max_index[3];  /* Index in each dimension of last
                                   * non-phoney */
    int            origin;      /* '0' or '1' */
    float          time;        /* Problem time */
    double         dtime;       /* Problem time, double data type */
   /*
    * The following field really only contains 3 elements.  However, silo
    * contains a bug in PJ_ReadVariable() as called by DBGetQuadvar() which
    * can cause three doubles to be stored there instead of three floats.
    */
    float          align[6];    /* Centering and alignment per dimension */

    float        **mixvals;     /* nvals ptrs to data arrays for mixed zones */
    int            mixlen;      /* Num of elmts in each mixed zone data
                                 * array */

    int            use_specmf;  /* Flag indicating whether to apply species
                                 * mass fractions to the variable. */

    int            ascii_labels;/* Treat variable values as ASCII values
                                   by rounding to the nearest integer in
                                   the range [0, 255] */
} DBquadvar;

typedef struct {
/*----------- Unstructured Cell Data (UCD) Variable -----------*/
    int            id;          /* Identifier for this object */
    char          *name;        /* Name of variable */
    int            cycle;       /* Problem cycle number */
    char          *units;       /* Units for variable, e.g, 'mm/ms' */
    char          *label;       /* Label (perhaps for editing purposes) */
    float          time;        /* Problem time */
    double         dtime;       /* Problem time, double data type */
    int            meshid;      /* Identifier for associated mesh */

    float        **vals;        /* Array of pointers to data arrays */
    int            datatype;    /* Type of data pointed to by 'vals' */
    int            nels;        /* Number of elements in each array */
    int            nvals;       /* Number of arrays pointed to by 'vals' */
    int            ndims;       /* Rank of variable */
    int            origin;      /* '0' or '1' */

    int            centering;   /* Centering within mesh (nodal or zonal) */
    float        **mixvals;     /* nvals ptrs to data arrays for mixed zones */
    int            mixlen;      /* Num of elmts in each mixed zone data
                                 * array */

    int            use_specmf;  /* Flag indicating whether to apply species
                                 * mass fractions to the variable. */
    int            ascii_labels;/* Treat variable values as ASCII values
                                   by rounding to the nearest integer in
                                   the range [0, 255] */
} DBucdvar;

typedef struct {
/*----------- Generic Mesh-Data Variable -----------*/
    int            id;          /* Identifier for this object */
    char          *name;        /* Name of variable */
    char          *units;       /* Units for variable, e.g, 'mm/ms' */
    char          *label;       /* Label (perhaps for editing purposes) */
    int            cycle;       /* Problem cycle number */
    int            meshid;      /* Identifier for associated mesh */

    float        **vals;        /* Array of pointers to data arrays */
    int            datatype;    /* Type of data pointed to by 'val' */
    int            nels;        /* Number of elements in each array */
    int            nvals;       /* Number of arrays pointed to by 'vals' */
    int            nspace;      /* Spatial rank of variable */
    int            ndims;       /* Rank of 'vals' array(s) (computatnl rank) */

    int            origin;      /* '0' or '1' */
    int            centering;   /* Centering within mesh (nodal,zonal,other) */
    float          time;        /* Problem time */
    double         dtime;       /* Problem time, double data type */
   /*
    * The following field really only contains 3 elements.  However, silo
    * contains a bug in PJ_ReadVariable() as called by DBGetPointvar() which
    * can cause three doubles to be stored there instead of three floats.
    */
    float          align[6];    /* Alignmnt per dimension if
                                 * centering==other */

    /* Stuff for multi-dimensional arrays (ndims > 1) */
    int            dims[3];     /* Number of elements in each dimension */
    int            major_order; /* 1 indicates row-major for multi-d arrays */
    int            stride[3];   /* Offsets to adjacent elements  */
   /*
    * The following two fields really only contain 3 elements.  However, silo
    * contains a bug in PJ_ReadVariable() as called by DBGetUcdmesh() which
    * can cause three doubles to be stored there instead of three floats.
    */
    int            min_index[6];  /* Index in each dimension of 1st
                                   * non-phoney */
    int            max_index[6];  /* Index in each dimension of last
                                    non-phoney */

    int            ascii_labels;/* Treat variable values as ASCII values
                                   by rounding to the nearest integer in
                                   the range [0, 255] */
} DBmeshvar;

typedef struct {
/*----------- Material Information -----------*/
    int            id;          /* Identifier */
    char          *name;        /* Name of this material information block */
    int            ndims;       /* Rank of 'matlist' variable */
    int            origin;      /* '0' or '1' */
    int            dims[3];     /* Number of elements in each dimension */
    int            major_order; /* 1 indicates row-major for multi-d arrays */
    int            stride[3];   /* Offsets to adjacent elements in matlist */

    int            nmat;        /* Number of materials */
    int           *matnos;      /* Array [nmat] of valid material numbers */
    char         **matnames;    /* Array of material names   */
    int           *matlist;     /* Array[nzone] w/ mat. number or mix index */
    int            mixlen;      /* Length of mixed data arrays (mix_xxx) */
    int            datatype;    /* Type of volume-fractions (double,float) */
    float         *mix_vf;      /* Array [mixlen] of volume fractions */
    int           *mix_next;    /* Array [mixlen] of mixed data indeces */
    int           *mix_mat;     /* Array [mixlen] of material numbers */
    int           *mix_zone;    /* Array [mixlen] of back pointers to mesh */
} DBmaterial;

typedef struct {
/*----------- Species Information -----------*/
    int            id;          /* Identifier */
    char          *name;        /* Name of this matspecies information block */
    char          *matname;     /* Name of material object with which the
                                 * material species object is associated. */
    int            nmat;        /* Number of materials */
    int           *nmatspec;    /* Array of lngth nmat of the num of material
                                 * species associated with each material. */
    int            ndims;       /* Rank of 'speclist' variable */
    int            dims[3];     /* Number of elements in each dimension of the
                                 * 'speclist' variable. */
    int            major_order; /* 1 indicates row-major for multi-d arrays */
    int            stride[3];   /* Offsts to adjacent elmts in 'speclist'  */

    int            nspecies_mf; /* Total number of species mass fractions. */
    float         *species_mf;  /* Array of length nspecies_mf of mass
                                 * frations of the material species. */
    int           *speclist;    /* Zone array of dimensions described by ndims
                                 * and dims.  Each element of the array is an
                                 * index into one of the species mass fraction
                                 * arrays.  A positive value is the index in
                                 * the species_mf array of the mass fractions
                                 * of the clean zone's material species:
                                 * species_mf[speclist[i]] is the mass fraction
                                 * of the first species of material matlist[i]
                                 * in zone i. A negative value means that the
                                 * zone is a mixed zone and that the array
                                 * mix_speclist contains the index to the
                                 * species mas fractions: -speclist[i] is the
                                 * index in the 'mix_speclist' array for zone
                                 * i. */
    int            mixlen;      /* Length of 'mix_speclist' array. */
    int           *mix_speclist;  /* Array of lgth mixlen of 1-orig indices
                                   * into the 'species_mf' array.
                                   * species_mf[mix_speclist[j]] is the index
                                   * in array species_mf' of the first of the
                                   * mass fractions for material
                                   * mix_mat[j]. */

    int            datatype;    /* Datatype of mass fraction data. */
} DBmatspecies;

/*-------------------------------------------------------------------------
 * A compound array is an array whose elements are simple arrays. A simple
 * array is an array whose elements are all of the same primitive data
 * type: float, double, integer, long...  All of the simple arrays of
 * a compound array have elements of the same data type.
 *-------------------------------------------------------------------------
 */
typedef struct {
    int            id;          /*identifier of the compound array */
    char          *name;        /*name of te compound array  */
    char         **elemnames;   /*names of the simple array elements */
    int           *elemlengths; /*lengths of the simple arrays  */
    int            nelems;      /*number of simple arrays  */
    void          *values;      /*simple array values   */
    int            nvalues;     /*sum reduction of `elemlengths' vector */
    int            datatype;    /*simple array element data type */
} DBcompoundarray;

typedef struct {

    int           *options;     /* Vector of option identifiers */
    void         **values;      /* Vector of pointers to option values */
    int            numopts;     /* Number of options defined */
    int            maxopts;     /* Total length of option/value arrays */

} DBoptlist;

typedef struct {

    char          *name;
    char          *type;        /* Type of group/object */
    char         **comp_names;  /* Array of component names */
    char         **pdb_names;   /* Array of internal (PDB) variable names */
    int            ncomponents; /* Number of components */
    int            maxcomponents;  /* Max number of components */

} DBobject;

typedef struct DBfile *___DUMMY_TYPE;  /* Satisfy ANSI scope rules */

/*
 * All file formats are now anonymous except for the public properties
 * and public methods.
 */
typedef struct DBfile_pub {

    /* Public Properties */
    char          *name;        /*name of file    */
    int            type;        /*file type    */
    DBtoc         *toc;         /*table of contents   */
    int            dirid;       /*directory ID    */
    int            fileid;      /*unique file id [0,DB_NFILES-1] */
    int            pathok;      /*driver handles paths in names */

    /* Public Methods */
    int            (*close)(struct DBfile *);
    int            (*exist)(struct DBfile *, char *);
    int            (*pause)(struct DBfile *);
    int            (*cont)(struct DBfile *);
    int            (*newtoc)(struct DBfile *);
    DBObjectType   (*inqvartype)(struct DBfile *, char*);
    int            (*uninstall)(struct DBfile *);
    DBobject      *(*g_obj)(struct DBfile *, char *);
    int            (*c_obj)(struct DBfile *, DBobject *, int);
    int            (*w_obj)(struct DBfile *, DBobject *, int);
    void          *(*g_comp)(struct DBfile *, char *, char *);
    int            (*g_comptyp)(struct DBfile *, char *, char *);
    int            (*w_comp)(struct DBfile *, DBobject *, char *, char *,
                             char *, void *, int, long *);
    int            (*write) (struct DBfile *, char *, void *, int *, int, int);
    int            (*writeslice)(struct DBfile *, char *, void *, int,
                                 int[], int[], int[], int[], int);
    void          *(*g_attr)(struct DBfile *, char *, char *);
    int            (*g_dir)(struct DBfile *, char *);
    int            (*mkdir)(struct DBfile *, char *);
    int            (*cd)(struct DBfile *, char *);
    int            (*cdid)(struct DBfile *, int);
    int            (*r_att)(struct DBfile *, char *, char *, void *);
    int            (*r_var)(struct DBfile *, char *, void *);
    int            (*r_var1)(struct DBfile *, char *, int, void *);
    int            (*module)(struct DBfile *, FILE *);
    int            (*r_varslice)(struct DBfile *, char *, int *, int *, int *,
                                 int, void *);
    int            (*g_compnames)(struct DBfile *, char *, char ***, char ***);
    DBcompoundarray *(*g_ca)(struct DBfile *, char *);
    DBcurve       *(*g_cu)(struct DBfile *, char *);
    DBmaterial    *(*g_ma)(struct DBfile *, char *);
    DBmatspecies  *(*g_ms)(struct DBfile *, char *);
    DBmultimesh   *(*g_mm)(struct DBfile *, char *);
    DBmultivar    *(*g_mv)(struct DBfile *, char *);
    DBmultimat    *(*g_mt)(struct DBfile *, char *);
    DBmultimatspecies *(*g_mms)(struct DBfile *, char *);
    DBpointmesh   *(*g_pm)(struct DBfile *, char *);
    DBmeshvar     *(*g_pv)(struct DBfile *, char *);
    DBquadmesh    *(*g_qm)(struct DBfile *, char *);
    DBquadvar     *(*g_qv)(struct DBfile *, char *);
    DBucdmesh     *(*g_um)(struct DBfile *, char *);
    DBucdvar      *(*g_uv)(struct DBfile *, char *);
    DBfacelist    *(*g_fl)(struct DBfile *, char *);
    DBzonelist    *(*g_zl)(struct DBfile *, char *);
    void          *(*g_var)(struct DBfile *, char *);
    int            (*g_varbl)(struct DBfile *, char *);  /*byte length */
    int            (*g_varlen)(struct DBfile *, char *);  /*nelems */
    int            (*g_vardims)(struct DBfile*, char*, int, int*); /*dims*/
    int            (*g_vartype)(struct DBfile *, char *);
    int            (*i_meshname)(struct DBfile *, char *, char *);
    int            (*i_meshtype)(struct DBfile *, char *);
    int            (*p_ca)(struct DBfile *, char *, char **, int *, int,
                           void *, int, int, DBoptlist *);
    int            (*p_cu)(struct DBfile *, char *, void *, void *, int, int,
                           DBoptlist *);
    int            (*p_fl)(struct DBfile *, char *, int, int, int *, int, int,
                           int *, int *, int *, int, int *, int *, int);
    int            (*p_ma)(struct DBfile *, char *, char *, int, int *, int *,
                           int *, int, int *, int *, int *, float *, int, int,
                           DBoptlist *);
    int            (*p_ms)(struct DBfile *, char *, char *, int, int *, int *,
                           int *, int, int, float *, int *, int, int,
                           DBoptlist *);
    int            (*p_mm)(struct DBfile *, char *, int, char **, int *,
                           DBoptlist *);
    int            (*p_mv)(struct DBfile *, char *, int, char **, int *,
                           DBoptlist *);
    int            (*p_mt)(struct DBfile *, char *, int, char **, DBoptlist *);
    int            (*p_mms)(struct DBfile *, char *, int, char **, DBoptlist *);
    int            (*p_pm)(struct DBfile *, char *, int, float **, int, int,
                           DBoptlist *);
    int            (*p_pv)(struct DBfile *, char *, char *, int, float **, int,
                           int, DBoptlist *);
    int            (*p_qm)(struct DBfile *, char *, char **, float **, int *,
                           int, int, int, DBoptlist *);
    int            (*p_qv)(struct DBfile *, char *, char *, int, char **,
                           float **, int *, int, float **, int, int, int,
                           DBoptlist *);
    int            (*p_um)(struct DBfile *, char *, int, char **, float **,
                           int, int, char *, char *, int, DBoptlist *);
    int            (*p_sm)(struct DBfile *, char *, char *,
                           int, char *, char *, DBoptlist *);
    int            (*p_uv)(struct DBfile *, char *, char *, int, char **,
                           float **, int, float **, int, int, int,
                           DBoptlist *);
    int            (*p_zl)(struct DBfile *, char *, int, int, int *, int, int,
                           int *, int *, int);
    int            (*p_zl2)(struct DBfile *, char *, int, int, int *, int, int,
                            int, int, int *, int *, int *, int, DBoptlist *);
    /* MCM-27Jul04: We added these to the end to avert potential
       link-time compatibility issues with older versions of the
       library. Some user's of Silo circumvent its version check
       which would ordinarily keep different versions from being
       mixed by defining an appropriate global symbol. */
    DBphzonelist  *(*g_phzl)(struct DBfile *, char *);
    int            (*p_phzl)(struct DBfile *, char *,
                             int, int *, int, int *, char *,
                             int, int *, int, int *,
                             int, int, int,
                             DBoptlist *);

} DBfile_pub;

typedef struct DBfile {
    DBfile_pub     pub;
    /*private part follows per device driver */
} DBfile;


/* The first prototypes here are the functions by which client code first
 * gets into Silo.  They are separated out because they do a version number
 * check for us.  Client code doesn't actually use these functions.
 * Instead, it uses macros like DBOpen, DBCreate, etc.
 *
 * If any functions are added that provide first-call access to Silo, they
 * should be set up as macro/function pairs, just as these are.  The way to
 * determine if a function is a "first-call" function is to ask whether
 * there are any Silo calls that must happen before it.  If there are not,
 * then the function is a "first-call" function and should have this
 * macro/function pair.  */

SILO_API extern DBfile  *DBOpenReal(const char *, int, int);
SILO_API extern DBfile  *DBCreateReal(const char *, int, int, const char *, int);
SILO_API extern int      DBInqFileReal(const char *);

#define SILO_VERSION Silo_version_4_4_3
SILO_API extern int SILO_VERSION;
#define CheckVersion SILO_VERSION = 1

#define DBOpen(name, target, mode) \
    (CheckVersion, DBOpenReal(name, target, mode))

#define DBCreate(name, mode, target, info, type) \
    (CheckVersion, DBCreateReal(name, mode, target, info, type))

#define DBInqFile(name) \
    (CheckVersion, DBInqFileReal(name))

/* Prototypes for regular API functions. */
SILO_API extern DBcompoundarray *DBAllocCompoundarray(void);
SILO_API extern DBcurve *DBAllocCurve(void);
SILO_API extern DBmultimesh *DBAllocMultimesh(int);
SILO_API extern DBmultivar *DBAllocMultivar(int);
SILO_API extern DBmultimat *DBAllocMultimat(int);
SILO_API extern DBmultimatspecies *DBAllocMultimatspecies(int);
SILO_API extern DBquadmesh *DBAllocQuadmesh(void);
SILO_API extern DBpointmesh *DBAllocPointmesh(void);
SILO_API extern DBmeshvar *DBAllocMeshvar(void);
SILO_API extern DBucdmesh *DBAllocUcdmesh(void);
SILO_API extern DBquadvar *DBAllocQuadvar(void);
SILO_API extern DBucdvar *DBAllocUcdvar(void);
SILO_API extern DBzonelist *DBAllocZonelist(void);
SILO_API extern DBphzonelist *DBAllocPHZonelist(void);
SILO_API extern DBedgelist *DBAllocEdgelist(void);
SILO_API extern DBfacelist *DBAllocFacelist(void);
SILO_API extern DBmaterial *DBAllocMaterial(void);
SILO_API extern DBmatspecies *DBAllocMatspecies(void);

SILO_API extern void     DBFreeMatspecies(DBmatspecies *);
SILO_API extern void     DBFreeMaterial(DBmaterial *);
SILO_API extern void     DBFreeFacelist(DBfacelist *);
SILO_API extern void     DBFreeEdgelist(DBedgelist *);
SILO_API extern void     DBFreeZonelist(DBzonelist *);
SILO_API extern void     DBFreePHZonelist(DBphzonelist *);
SILO_API extern void     DBResetUcdvar(DBucdvar *);
SILO_API extern void     DBFreeUcdvar(DBucdvar *);
SILO_API extern void     DBResetQuadvar(DBquadvar *);
SILO_API extern void     DBFreeQuadvar(DBquadvar *);
SILO_API extern void     DBFreeUcdmesh(DBucdmesh *);
SILO_API extern void     DBFreeMeshvar(DBmeshvar *);
SILO_API extern void     DBFreePointmesh(DBpointmesh *);
SILO_API extern void     DBFreeQuadmesh(DBquadmesh *);
SILO_API extern void     DBFreeMultimesh(DBmultimesh *);
SILO_API extern void     DBFreeMultivar(DBmultivar *);
SILO_API extern void     DBFreeMultimat(DBmultimat *);
SILO_API extern void     DBFreeMultimatspecies(DBmultimatspecies *);
SILO_API extern void     DBFreeCompoundarray(DBcompoundarray *);
SILO_API extern void     DBFreeCurve(DBcurve *);

SILO_API extern long     DBSetDataReadMask(long);
SILO_API extern long     DBGetDataReadMask(void);
SILO_API extern char    *DBVersion(void);
SILO_API extern void     DBShowErrors(int, void (*)(char *));
SILO_API extern char    *DBErrString(void);
SILO_API extern char    *DBErrFunc(void);
SILO_API extern int      DBClose(DBfile *);
SILO_API extern int      DBPause(DBfile *);
SILO_API extern int      DBContinue(DBfile *);
SILO_API extern int      DBInqVarExists(DBfile *, const char *);
SILO_API extern int      DBForceSingle(int);
SILO_API extern int      DBUninstall(DBfile *);
SILO_API extern DBoptlist *DBMakeOptlist(int);
SILO_API extern int      DBClearOptlist(DBoptlist *);
SILO_API extern int      DBFreeOptlist(DBoptlist *);
SILO_API extern int      DBAddOption(DBoptlist *, int, void *);
SILO_API extern DBtoc   *DBGetToc(DBfile *);
SILO_API extern int      DBNewToc(DBfile *);
SILO_API extern int      DBFilters(DBfile *, FILE *);
SILO_API extern int      DBFilterRegistration(const char *, int (*init) (DBfile *, char *),
                                     int (*open) (DBfile *, char *));
SILO_API extern void    *DBGetAtt(DBfile *, const char *, const char *);
SILO_API extern DBobject *DBGetObject(DBfile *, const char *);
SILO_API extern int      DBChangeObject(DBfile *, DBobject *);
SILO_API extern int      DBWriteObject(DBfile *, DBobject *, int);
SILO_API extern void    *DBGetComponent(DBfile *, const char *, const char *);
SILO_API extern int      DBGetComponentType(DBfile *, const char *, const char *);
SILO_API extern int      DBWriteComponent(DBfile *, DBobject *, const char *, const char *, const char *,
                                 void *, int, long *);
SILO_API extern int      DBWrite(DBfile *, const char *, void *, int *, int, int);
SILO_API extern int      DBWriteSlice(DBfile *, const char *, void *, int, int[], int[],
                             int[], int[], int);
SILO_API extern DBfacelist *DBCalcExternalFacelist(int *, int, int, int *, int *, int,
                                          int *, int);
SILO_API extern DBfacelist *DBCalcExternalFacelist2(int *, int, int, int, int, int *,
                                           int *, int *, int, int *, int);
SILO_API extern int      DBGetDir(DBfile *, char *);
SILO_API extern int      DBSetDir(DBfile *, const char *);
SILO_API extern int      DBSetDirID(DBfile *, int);
SILO_API extern int      DBListDir(DBfile *, char **, int);
SILO_API extern int      DBMkDir(DBfile *, const char *);

#define DBMkdir DBMkDir
SILO_API extern int      DBReadAtt(DBfile *, const char *, const char *, void *);
SILO_API extern int      DBRead(DBfile *, const char *, void *);
SILO_API extern int      DBReadVar(DBfile *, const char *, void *);
SILO_API extern int      DBReadVar1(DBfile *, const char *, int, void *);
SILO_API extern int      DBReadVarSlice(DBfile *, const char *, int *, int *, int *, int,
                               void *);
SILO_API extern DBobject *DBMakeObject(const char *, int, int);
SILO_API extern int      DBFreeObject(DBobject *);
SILO_API extern int      DBClearObject(DBobject *);
SILO_API extern int      DBAddVarComponent(DBobject *, const char *, const char *);
SILO_API extern int      DBAddIntComponent(DBobject *, const char *, int);
SILO_API extern int      DBAddFltComponent(DBobject *, const char *, double);
SILO_API extern int      DBAddDblComponent(DBobject *, const char *, double);
SILO_API extern int      DBAddStrComponent(DBobject *, const char *, const char *);
SILO_API extern int      DBGetComponentNames(DBfile *, const char *, char ***, char ***);

SILO_API extern DBcompoundarray *DBGetCompoundarray(DBfile *, const char *);
SILO_API extern DBcurve *DBGetCurve(DBfile *, const char *);
SILO_API extern DBmaterial *DBGetMaterial(DBfile *, const char *);
SILO_API extern DBmatspecies *DBGetMatspecies(DBfile *, const char *);
SILO_API extern DBmultimesh *DBGetMultimesh(DBfile *, const char *);
SILO_API extern DBmultivar *DBGetMultivar(DBfile *, const char *);
SILO_API extern DBmultimat *DBGetMultimat(DBfile *, const char *);
SILO_API extern DBmultimatspecies *DBGetMultimatspecies(DBfile *, const char *);
SILO_API extern DBpointmesh *DBGetPointmesh(DBfile *, const char *);
SILO_API extern DBmeshvar *DBGetPointvar(DBfile *, const char *);
SILO_API extern DBquadmesh *DBGetQuadmesh(DBfile *, const char *);
SILO_API extern DBquadvar *DBGetQuadvar(DBfile *, const char *);
SILO_API extern int      DBGetQuadvar1(DBfile *, const char *, float *, int *, int *,
                              float *, int *, int *, int *);
SILO_API extern int      DBAnnotateUcdmesh(DBucdmesh *);
SILO_API extern DBucdmesh *DBGetUcdmesh(DBfile *, const char *);
SILO_API extern DBucdvar *DBGetUcdvar(DBfile *, const char *);
SILO_API extern DBfacelist *DBGetFacelist(DBfile *, const char *);
SILO_API extern DBzonelist *DBGetZonelist(DBfile *, const char *);
SILO_API extern DBphzonelist *DBGetPHZonelist(DBfile *, const char *);
SILO_API extern void    *DBGetVar(DBfile *, const char *);
SILO_API extern int      DBGetVarByteLength(DBfile *, const char *);
SILO_API extern int      DBGetVarLength(DBfile *, const char *);
SILO_API extern int      DBGetVarDims(DBfile *, const char *, int, int *);
SILO_API extern int      DBGetVarType(DBfile *, const char *);
SILO_API extern int      DBInqMeshname(DBfile *, const char *, const char *);
SILO_API extern int      DBInqMeshtype(DBfile *, const char *);
SILO_API extern int      DBInqCompoundarray(DBfile *, const char *, char ***,
                                   int **, int *, int *, int *);
SILO_API extern DBObjectType DBInqVarType(DBfile *, const char *);

SILO_API extern int      DBPutCompoundarray(DBfile *, const char *, char **, int *, int,
                                   void *, int, int, DBoptlist *);
SILO_API extern int      DBPutCurve(DBfile *, const char *, void *, void *, int, int,
                           DBoptlist *);
SILO_API extern int      DBPutFacelist(DBfile *, const char *, int, int, int *, int, int,
                            int *, int *, int *, int, int *, int *, int);
SILO_API extern int      DBPutMaterial(DBfile *, const char *, const char *, int, int *, int *,
                           int *, int, int *, int *, int *, float *, int,
                              int, DBoptlist *);
SILO_API extern int      DBPutMatspecies(struct DBfile *, const char *, const char *, int, int *,
                                int *, int *, int, int, float *, int *, int, int,
                                DBoptlist *);
SILO_API extern int      DBPutMultimesh(DBfile *, const char *, int, char **, int *,
                               DBoptlist *);
SILO_API extern int      DBPutMultivar(DBfile *, const char *, int, char **, int *,
                              DBoptlist *);
SILO_API extern int      DBPutMultimat(DBfile *, const char *, int, char **, DBoptlist *);
SILO_API extern int      DBPutMultimatspecies(DBfile *, const char *, int, char **,
                                     DBoptlist *);
SILO_API extern int      DBPutPointmesh(DBfile *, const char *, int, float **, int, int,
                               DBoptlist *);
SILO_API extern int      DBPutPointvar(DBfile *, const char *, const char *, int, float **, int,
                              int, DBoptlist *);
SILO_API extern int      DBPutPointvar1(DBfile *, const char *, const char *, float *, int, int,
                               DBoptlist *);
SILO_API extern int      DBPutQuadmesh(DBfile *, const char *, char **, float **, int *, int,
                              int, int, DBoptlist *);
SILO_API extern int      DBPutQuadvar(DBfile *, const char *, const char *, int, char **, float **,
                             int *, int, float **, int, int, int, DBoptlist *);
SILO_API extern int      DBPutQuadvar1(DBfile *, const char *, const char *, float *, int *, int,
                              float *, int, int, int, DBoptlist *);
SILO_API extern int      DBPutUcdmesh(DBfile *, const char *, int, char **, float **, int,
                             int, const char *, const char *, int, DBoptlist *);
SILO_API extern int      DBPutUcdsubmesh(DBfile *, const char *, const char *, int,
                             const char *, const char *, DBoptlist *);
SILO_API extern int      DBPutUcdvar(DBfile *, const char *, const char *, int, char **, float **,
                            int, float **, int, int, int, DBoptlist *);
SILO_API extern int      DBPutUcdvar1(DBfile *, const char *, const char *, float *, int, float *,
                             int, int, int, DBoptlist *);
SILO_API extern int      DBPutZonelist(DBfile *, const char *, int, int, int *, int, int,
                              int *, int *, int);
SILO_API extern int      DBPutZonelist2(DBfile *, const char *, int, int, int *, int, int,
                               int, int, int *, int *, int *, int, DBoptlist*);
SILO_API extern int      DBPutPHZonelist(DBfile *, const char *, int, int *, int, int *, const char *,
                                                           int, int *, int, int *,
                                                           int, int, int, DBoptlist *);
SILO_API extern void *   DBFortranAccessPointer(int value);
SILO_API extern int      DBFortranAllocPointer(void *pointer);
SILO_API extern void     DBFortranRemovePointer(int value);

/*-------------------------------------------------------------------------
 * Public global variables.
 *-------------------------------------------------------------------------
 */
SILO_API extern int     DBDebugAPI;      /*file desc for debug messages, or zero */
SILO_API extern int     db_errno;        /*error number of last error */
SILO_API extern char    db_errfunc[];    /*name of erring function */

#ifndef DB_MAIN
SILO_API extern DBfile *(*DBOpenCB[])(const char *, int);
SILO_API extern DBfile *(*DBCreateCB[])(const char *, int, int, const char *);
SILO_API extern int     (*DBFSingleCB[])(int);
#endif

#ifdef __cplusplus
}
#endif
#undef NO_FORTRAN_DEFINE
#endif /* !SILO_H */
