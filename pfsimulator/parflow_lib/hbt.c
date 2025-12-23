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

#include "hbt.h"

#include <stdlib.h>
#include <string.h>

/*===========================================================================*/
/* The stack size for the deletion algorithm.                                */
/*===========================================================================*/
#define STACK_SIZE 100

/*===========================================================================*/
/* Each node has flag indicating whether it is balanced or not.              */
/* NOTE: comparisons are done using ">" and "<" so sign is important         */
/*===========================================================================*/
#define BALANCED 0
#define UNBALANCED_LEFT -1
#define UNBALANCED_RIGHT 1

/*===========================================================================*/
/* Some macros for easy access to fields in the HBT structure            */
/*===========================================================================*/
#define LEFT(e) (e)->left
#define RIGHT(e) (e)->right
#define B(e) (e)->balance
#define OBJ(e) (e)->obj


/*===========================================================================*/
/* Creates a new HBT.                                                    */
/* The compare method function has form:                                     */
/*    int compare(void *a, void *b)                                          */
/* returns <0 if a < b, 0 if a=b, >0 if a > b                                */
/*===========================================================================*/
HBT *HBT_new(
             int (*compare_method)(void *, void *),
             void (*free_method)(void *),
             void (*printf_method)(FILE *, void *),
             int (*scanf_method)(FILE *, void **),
             int malloc_flag)
{
  HBT *new_tree;

  if ((new_tree = (HBT*)calloc(1, sizeof(HBT))) == NULL)
    return NULL;

  new_tree->compare = compare_method;
  new_tree->free = free_method;
  new_tree->printf = printf_method;
  new_tree->scanf = scanf_method;
  new_tree->malloc_flag = malloc_flag;

  return new_tree;
}

/*===========================================================================*/
/* Creates a new HBT element.                                            */
/*===========================================================================*/
HBT_element *_new_HBT_element(
                              HBT * tree,
                              void *object,
                              int   sizeof_obj)
{
  HBT_element *n;

  if ((n = (HBT_element*)calloc(1, sizeof(HBT_element))) == NULL)
    return NULL;

  /* how we add depends on flag */
  if (tree->malloc_flag)
    if ((n->obj = (void*)calloc(1, sizeof_obj)) != NULL)
      memcpy(n->obj, object, sizeof_obj);
    else
    {
      free(n);
      return NULL;
    }
  else
    n->obj = object;

  return n;
}

/*===========================================================================*/
/* Frees a HBT element.                                                  */
/*===========================================================================*/
void _free_HBT_element(
                       HBT *        tree,
                       HBT_element *el)
{
  if (tree->free)
    (*tree->free)(OBJ(el));

  if (tree->malloc_flag)
    free(OBJ(el));

  free(el);
}

/*===========================================================================*/
/* Frees elements of HBT recursively.                                    */
/*===========================================================================*/
void _HBT_free(
               HBT *        tree,
               HBT_element *subtree)
{
  if (subtree != NULL)
  {
    _HBT_free(tree, subtree->left);
    _HBT_free(tree, subtree->right);
    _free_HBT_element(tree, subtree);
  }
}

/*===========================================================================*/
/* Frees a HBT tree.                                                     */
/*===========================================================================*/
void HBT_free(
              HBT *tree)
{
  _HBT_free(tree, tree->root);      /* free up any nodes still on tree   */
  free(tree);
}


/*===========================================================================*/
/* Searches HBT for key.                                                 */
/*===========================================================================*/
void *HBT_lookup(
                 HBT * tree,
                 void *obj)
{
  int test;
  HBT_element *temp;

  temp = tree->root;

  while (temp)
    if ((test =
           (*tree->compare)(obj, OBJ(temp))) < 0)
      /*---------------------------------------------------------------*/
      /* Go left.                                                      */
      /*---------------------------------------------------------------*/
      temp = LEFT(temp);
    else if (test > 0)
      /*---------------------------------------------------------------*/
      /* Go right.                                                     */
      /*---------------------------------------------------------------*/
      temp = RIGHT(temp);
    else
      return OBJ(temp);
  /* return the object that matches the template
   * object the user gave us.                       */

  return NULL;
}

/*===========================================================================*/
/* Replaces HBT for key.                                                 */
/*===========================================================================*/
void *HBT_replace(
                  HBT * tree,
                  void *obj,
                  int   sizeof_obj)
{
  int test;
  int found = FALSE;
  HBT_element *temp;

  (void)sizeof_obj;

  temp = tree->root;

  while (temp)
    if ((test =
           (*tree->compare)(obj, OBJ(temp))) < 0)
    {
      /*---------------------------------------------------------------*/
      /* Go left.                                                      */
      /*---------------------------------------------------------------*/
      temp = LEFT(temp);
    }
    else if (test > 0)
    {
      /*---------------------------------------------------------------*/
      /* Go right.                                                     */
      /*---------------------------------------------------------------*/
      temp = RIGHT(temp);
    }
    else
      found = TRUE;
  /* Found the object that is to be replaced */

  if (found)
  {
    if (tree->malloc_flag)
      free(temp->obj);

    temp->obj = obj;

    return temp;
  }
  else
    return NULL;
}

/*===========================================================================*/
/* Inserts a new node into the HBT and rebalances the tree if it gets        */
/* unbalanced while adding.                                                  */
/*===========================================================================*/
int HBT_insert(
               HBT * tree,
               void *obj,
               int   sizeof_obj)
{
  HBT_element *temp, *inserted, *rebalance_son, *rebalance,
              *rebalance_father;
  int done = 0;
  int test, test_rebalance;
  short rebalance_B;

  int (*compare)(void *, void *) = tree->compare;

  /*-------------------------------------------------------------------------*/
  /* If tree is currently empty then just add new node.                      */
  /*-------------------------------------------------------------------------*/
  if (tree->root == NULL)
  {
    if ((tree->root = _new_HBT_element(tree, obj, sizeof_obj)) == NULL)
    {
      return HBT_MEMORY_ERROR;
    }

    tree->height = 1;
    tree->num = 1;
    return HBT_INSERTED;
  }

  /*-------------------------------------------------------------------------*/
  /* Initialize the starting location for balance adjustment and rebalancing */
  /* efforts.                                                                */
  /*-------------------------------------------------------------------------*/
  rebalance_father = (HBT_element*)tree;
  rebalance = temp = tree->root;

  /*-------------------------------------------------------------------------*/
  /* Find the place where the new node should go.                            */
  /*-------------------------------------------------------------------------*/
  while (!done)
  {
    if ((test = (*compare)(obj, OBJ(temp))) == 0)
    {
      /*-----------------------------------------------------------------*/
      /* The key already exists in the tree can't add.                   */
      /*-----------------------------------------------------------------*/
      return HBT_FOUND;
    }
    else if (test < 0)
    {
      /*-----------------------------------------------------------------*/
      /* Go left.                                                        */
      /*-----------------------------------------------------------------*/
      if ((inserted = LEFT(temp)) == NULL)
      {
        /*-------------------------------------------------------------*/
        /* Found place to insert the new node                          */
        /*-------------------------------------------------------------*/
        if ((inserted = (LEFT(temp) =
                           _new_HBT_element(tree, obj, sizeof_obj))) == NULL)
        {
          return HBT_MEMORY_ERROR;
        }
        done = 1;
      }
      else
      {
        /*-------------------------------------------------------------*/
        /* If not balanced at this point the move the starting location*/
        /* for rebalancing effort here.                                */
        /*-------------------------------------------------------------*/
        if (B(inserted) != BALANCED)
        {
          rebalance_father = temp;
          rebalance = inserted;
        }
        temp = inserted;
      }
    }
    else if (test > 0)
    {
      /*---------------------------------------------------------------*/
      /* Go left.                                                      */
      /*---------------------------------------------------------------*/

      if ((inserted = RIGHT(temp)) == NULL)
      {
        /*-------------------------------------------------------------*/
        /* Found place to insert the new node                          */
        /*-------------------------------------------------------------*/
        if ((inserted = (RIGHT(temp) =
                           _new_HBT_element(tree, obj, sizeof_obj))) == NULL)
        {
          return HBT_MEMORY_ERROR;
        }
        done = 1;
      }
      else
      {
        /*-------------------------------------------------------------*/
        /* If not balanced at this point the move the starting location*/
        /* for rebalancing effort here.                                */
        /*-------------------------------------------------------------*/
        if (B(inserted) != BALANCED)
        {
          rebalance_father = temp;
          rebalance = inserted;
        }
        temp = inserted;
      }
    }
  }

  /*-------------------------------------------------------------------------*/
  /* We have added another node so increase the number in tree.              */
  /*-------------------------------------------------------------------------*/
  tree->num++;


  /*-------------------------------------------------------------------------*/
  /* Adjust the balance factors along the path just traversed.  Only need to */
  /* do this on part of the path.                                            */
  /*-------------------------------------------------------------------------*/
  if ((test_rebalance = (*compare)(obj, OBJ(rebalance))) < 0)
    rebalance_son = temp = LEFT(rebalance);
  else
    rebalance_son = temp = RIGHT(rebalance);

  while (temp != inserted)
  {
    if ((test = (*compare)(obj, OBJ(temp))) < 0)
    {
      B(temp) = UNBALANCED_LEFT;
      temp = LEFT(temp);
    }
    else if (test > 0)
    {
      B(temp) = UNBALANCED_RIGHT;
      temp = RIGHT(temp);
    }
  }


  /*-------------------------------------------------------------------------*/
  /* Rebalence the tree.  There is only one point where rebalancing might    */
  /* be needed.                                                              */
  /*-------------------------------------------------------------------------*/
  rebalance_B = (test_rebalance < 0) ? UNBALANCED_LEFT : UNBALANCED_RIGHT;

  if (B(rebalance) == BALANCED)
  {
    /*---------------------------------------------------------------------*/
    /* Tree was balanced, adding new node simply unbalances it.            */
    /*---------------------------------------------------------------------*/
    B(rebalance) = rebalance_B;
    tree->height++;
  }
  else if (B(rebalance) == -rebalance_B)
    /*---------------------------------------------------------------------*/
    /* Tree was unbalanced towards the opposite side of insertion so       */
    /* with new node it is balanced.                                       */
    /*---------------------------------------------------------------------*/
    B(rebalance) = BALANCED;
  else
  {
    /*---------------------------------------------------------------------*/
    /* Tree needs rotated.                                                 */
    /* See Knuth or Reingold for picture of the rotations much easier and  */
    /* clearer than any word description.                                  */
    /*---------------------------------------------------------------------*/
    if (B(rebalance_son) == rebalance_B)
    {
      /*-----------------------------------------------------------------*/
      /* Single rotation.                                                */
      /*-----------------------------------------------------------------*/
      temp = rebalance_son;
      if (rebalance_B == UNBALANCED_LEFT)
      {
        LEFT(rebalance) = RIGHT(rebalance_son);
        RIGHT(rebalance_son) = rebalance;
      }
      else
      {
        RIGHT(rebalance) = LEFT(rebalance_son);
        LEFT(rebalance_son) = rebalance;
      }

      B(rebalance) = (B(rebalance_son) = BALANCED);
    }
    else if (B(rebalance_son) == -rebalance_B)
    {
      /*-----------------------------------------------------------------*/
      /* double rotation                                                 */
      /*-----------------------------------------------------------------*/

      if (rebalance_B == UNBALANCED_LEFT)
      {
        temp = RIGHT(rebalance_son);
        RIGHT(rebalance_son) = LEFT(temp);
        LEFT(temp) = rebalance_son;
        LEFT(rebalance) = RIGHT(temp);
        RIGHT(temp) = rebalance;
      }
      else
      {
        temp = LEFT(rebalance_son);
        LEFT(rebalance_son) = RIGHT(temp);
        RIGHT(temp) = rebalance_son;
        RIGHT(rebalance) = LEFT(temp);
        LEFT(temp) = rebalance;
      }

      if (B(temp) == rebalance_B)
      {
        B(rebalance) = (short)-rebalance_B;
        B(rebalance_son) = BALANCED;
      }
      else if (B(temp) == 0)
        B(rebalance) = (B(rebalance_son) = BALANCED);
      else
      {
        B(rebalance) = BALANCED;
        B(rebalance_son) = rebalance_B;
      }

      B(temp) = BALANCED;
    }

    /*---------------------------------------------------------------------*/
    /* Need to adjust what the father of the this subtree points at since  */
    /* we rotated.                                                         */
    /*---------------------------------------------------------------------*/
    if (rebalance_father == (HBT_element*)tree)
      tree->root = temp;
    else
    {
      if (rebalance == RIGHT(rebalance_father))
        RIGHT(rebalance_father) = temp;
      else
        LEFT(rebalance_father) = temp;
    }
  }

  return HBT_INSERTED;
}


/*---------------------------------------------------------------------------*/
/* Deletes an object from the tree.                                          */
/* If user is controlling allocation returns pointer to obj found.           */
/*---------------------------------------------------------------------------*/
void *HBT_delete(
                 HBT * tree,
                 void *obj)
{
  /*-----------------------------------------------------------------------*/
  /* The stack keeps elements from root to the node to be deleted.         */
  /* element_stack has pointers to the nodes.  dir_stack has the direction */
  /* taken down the path.                                                  */
  /*-----------------------------------------------------------------------*/
  int size = 0;
  HBT_element *element_stack[STACK_SIZE];
  short dir_stack[STACK_SIZE];
  short father_dir;
  void *ret;

  HBT_element *del, *successor, *father, *current, *son, *grandson, *temp;
  int test, done, top_of_stack;
  short dir;

  int (*compare)(void *, void *) = tree->compare;

  /*-----------------------------------------------------------------------*/
  /* Find the node to be deleted.  Keep track of path on the stack.        */
  /*-----------------------------------------------------------------------*/

  /*-----------------------------------------------------------------------*/
  /* put the HBT as indicator on the top of the stack.  This simplifies*/
  /* logic a little.                                                       */
  /*-----------------------------------------------------------------------*/
  element_stack[size++] = (HBT_element*)tree;
  element_stack[size] = (del = tree->root);

  while (del && (test = (*compare)(obj, OBJ(del))))
  {
    if (test < 0)
    {
      dir_stack[size] = UNBALANCED_LEFT;
      del = LEFT(del);
    }
    else
    {
      dir_stack[size] = UNBALANCED_RIGHT;
      del = RIGHT(del);
    }
    element_stack[++size] = del;
  }

  /*-----------------------------------------------------------------------*/
  /* Node was not found so can't delete it.                                */
  /*-----------------------------------------------------------------------*/
  if (del == NULL)
    return NULL;


  /*-----------------------------------------------------------------------*/
  /* Set father to be the parent of the node to be deleted                 */
  /*-----------------------------------------------------------------------*/
  father = element_stack[size - 1];
  father_dir = dir_stack[size - 1];

  /*-----------------------------------------------------------------------*/
  /* Do the actual deletion of the node from the tree.                     */
  /* Special processing is done if del is at the root of the tree.         */
  /*-----------------------------------------------------------------------*/
  if (RIGHT(del) == NULL)
  {
    /*-------------------------------------------------------------------*/
    /* RIGHT of del is null so we can just delete the node               */
    /*-------------------------------------------------------------------*/

    /* set dir to indicate where the NULL child is                       */
    dir_stack[size] = UNBALANCED_RIGHT;

    if (father == (HBT_element*)tree)
    {
      tree->root = LEFT(del);
      tree->height--;
    }
    else

    if (father_dir < 0)
      LEFT(father) = LEFT(del);
    else
      RIGHT(father) = LEFT(del);
  }
  else if (LEFT(del) == NULL)
  {
    /*-------------------------------------------------------------------*/
    /* LEFT of del is null so we can just delete the node                */
    /*-------------------------------------------------------------------*/

    /* set dir to indicate where the NULL child is */
    dir_stack[size] = UNBALANCED_LEFT;

    if (father == (HBT_element*)tree)
    {
      tree->root = RIGHT(del);
      tree->height--;
    }
    else

    if (father_dir < 0)
      LEFT(father) = RIGHT(del);
    else
      RIGHT(father) = RIGHT(del);
  }
  else
  {
    /*-------------------------------------------------------------------*/
    /* The "trick" employed here is finding the successor to del with a  */
    /* left link that is NULL.  This successor node is then swapped with */
    /* the node that we want to delete.  Thus the number of cases for    */
    /* actual deletion are small.  The tree is out of order (del has     */
    /* had been placed behind successor) but this does not matter        */
    /* since the tree is not accessed during deletion and the del        */
    /* node will be deleted anyway.  The swapping process just moves     */
    /* successor to del position; del is not reinserted since it is to be*/
    /* removed.                                                          */
    /*-------------------------------------------------------------------*/
    temp = RIGHT(del);
    if (LEFT(temp) == NULL)
    {
      /*---------------------------------------------------------------*/
      /* This is a special case when the successor is the son of del.  */
      /*---------------------------------------------------------------*/

      /*---------------------------------------------------------------*/
      /* Need to fix the stack since del and successor have swapped.   */
      /*---------------------------------------------------------------*/
      dir_stack[size] = UNBALANCED_RIGHT;
      element_stack[size++] = temp;

      /*---------------------------------------------------------------*/
      /* Here is the swap of del and successor.                        */
      /*---------------------------------------------------------------*/
      LEFT(temp) = LEFT(del);
      if (father == (HBT_element*)tree)
        tree->root = temp;
      else
      if (father_dir < 0)
        LEFT(father) = temp;
      else
        RIGHT(father) = temp;

      /*---------------------------------------------------------------*/
      /* successor will be rebalanced but it would have had same       */
      /* balance as del with del in successors place                   */
      /*---------------------------------------------------------------*/
      B(temp) = B(del);
    }
    else
    {
      /*---------------------------------------------------------------*/
      /* Successor is not the son of del so a search must be done.     */
      /*---------------------------------------------------------------*/

      /*---------------------------------------------------------------*/
      /* Need to fix the stack since del and successor have swapped.   */
      /*---------------------------------------------------------------*/
      dir_stack[size] = UNBALANCED_RIGHT;
      element_stack[size++] = temp;

      /*---------------------------------------------------------------*/
      /* Find the first successor to del that has no left subtree.     */
      /*---------------------------------------------------------------*/
      successor = LEFT(temp);
      while ((LEFT(successor) != NULL))
      {
        dir_stack[size] = UNBALANCED_LEFT;
        element_stack[size++] = successor;
        successor = LEFT(successor);
      }
      ;


      /*---------------------------------------------------------------*/
      /* Here is the swap of del and successor.                        */
      /*---------------------------------------------------------------*/
      LEFT(successor) = LEFT(del);
      LEFT(element_stack[size - 1]) = RIGHT(successor);
      RIGHT(successor) = RIGHT(del);
      if (father == (HBT_element*)tree)
        tree->root = successor;
      else
      if (father_dir < 0)
        LEFT(father) = successor;
      else
        RIGHT(father) = successor;

      /*---------------------------------------------------------------*/
      /* successor will be rebalanced but it would have had same       */
      /* balance as del with del in successors place                   */
      /*---------------------------------------------------------------*/
      B(successor) = B(del);
    }
  }


  /*-----------------------------------------------------------------------*/
  /* Rebalance the tree.  Search up the stack that was kept and rebalance  */
  /* at each node if needed.  The search can be terminated if the subtree  */
  /* height has not changed; the balance of higher noded could not have    */
  /* changed.                                                              */
  /*-----------------------------------------------------------------------*/
  done = FALSE;

  /* NOTE that the element 0 is the tree HBT so we don't visit it */
  for (top_of_stack = size - 1; (top_of_stack > 0) && (!done);
       top_of_stack--)
  {
    current = element_stack[top_of_stack];
    dir = dir_stack[top_of_stack];

    if (B(current) == BALANCED)
    {
      /*-----------------------------------------------------------*/
      /* The subtree was balanced at this point.                   */
      /* Unbalance it since a node was deleted.  Since the height  */
      /* of this subtree was not changed we are done               */
      /*-----------------------------------------------------------*/
      if (dir == UNBALANCED_LEFT)
        B(current) = UNBALANCED_RIGHT;
      else
        B(current) = UNBALANCED_LEFT;

      done = TRUE;
    }
    else if (B(current) == dir)
    {
      /*-----------------------------------------------------------*/
      /* The subtree was unbalanced toward the side the deletion   */
      /* occurred on so the new subtree is balanced but it has     */
      /* height one less than before so the rebalencing must       */
      /* continue                                                  */
      /*-----------------------------------------------------------*/
      B(current) = BALANCED;

      if (top_of_stack == 1)
        tree->height--;
    }
    else
    {
      /*-----------------------------------------------------------*/
      /* The del node was on the unbalanced side so the subtree    */
      /* is unbalanced by two.  Need to do a rotation.             */
      /*-----------------------------------------------------------*/


      /*-----------------------------------------------------------*/
      /* The rotation that needs to be done can be determined from */
      /* the son of del.  Again referring to Knuth or Reingold     */
      /* would be more valuable than any written description that  */
      /* I could write.  One day, perhaps, we can include pictures */
      /* in comments.                                              */
      /*-----------------------------------------------------------*/
      if (dir == UNBALANCED_LEFT)
        son = RIGHT(current);
      else
        son = LEFT(current);

      if (B(son) == BALANCED)
      {
        /*-------------------------------------------------------*/
        /* Son was balanced do a single rotation.                */
        /* Since the subtree at father has not changed in        */
        /* height we are done.                                   */
        /*-------------------------------------------------------*/
        if (dir == UNBALANCED_LEFT)
        {
          RIGHT(current) = LEFT(son);
          LEFT(son) = current;
        }
        else
        {
          LEFT(current) = RIGHT(son);
          RIGHT(son) = current;
        }

        B(son) = dir;

        done = TRUE;
      }
      else if (B(son) == -dir)
      {
        /*-------------------------------------------------------*/
        /* son is balanced the opposite direction we             */
        /* took at current.  Need to reblance and continue       */
        /* since the tree is one shorter than before.            */
        /*-------------------------------------------------------*/

        if (dir == UNBALANCED_LEFT)
        {
          RIGHT(current) = LEFT(son);
          LEFT(son) = current;
        }
        else
        {
          LEFT(current) = RIGHT(son);
          RIGHT(son) = current;
        }

        /* current and son are balanced now */
        B(current) = (B(son) = BALANCED);

        if (top_of_stack == 1)
          tree->height--;
      }
      else
      {
        /*-------------------------------------------------------*/
        /* son is balanced the same direction we took at current */
        /* Need to do a double rotation and continue.            */
        /*-------------------------------------------------------*/
        if (dir == UNBALANCED_LEFT)
        {
          grandson = LEFT(son);
          RIGHT(current) = LEFT(grandson);
          LEFT(son) = RIGHT(grandson);
          LEFT(grandson) = current;
          RIGHT(grandson) = son;
        }
        else
        {
          grandson = RIGHT(son);
          LEFT(current) = RIGHT(grandson);
          RIGHT(son) = LEFT(grandson);
          RIGHT(grandson) = current;
          LEFT(grandson) = son;
        }

        /* adjust the balance factors */
        if (B(grandson) == BALANCED)
          B(son) = (B(current) = BALANCED);
        else if (B(grandson) == dir)
        {
          B(son) = (short)-dir;
          B(current) = BALANCED;
        }
        else
        {
          B(son) = BALANCED;
          B(current) = dir;
        }

        if (top_of_stack == 1)
          tree->height--;

        /*-------------------------------------------------------*/
        /* Double rotation puts grandson at root of subtree.     */
        /*-------------------------------------------------------*/
        son = grandson;
      }

      /*-----------------------------------------------------------*/
      /* Since we rotated the subtree has a new root; point father */
      /* of del at this new root.                                  */
      /*-----------------------------------------------------------*/
      if ((father = element_stack[top_of_stack - 1]) ==
          (HBT_element*)tree)
      {
        tree->root = son;
      }
      else
      {
        if (dir_stack[top_of_stack - 1] == UNBALANCED_LEFT)
          LEFT(father) = son;
        else
          RIGHT(father) = son;
      }
    }
  }

  /*-------------------------------------------------------------------*/
  /* Delete the element del                                            */
  /*-------------------------------------------------------------------*/

  if (tree->malloc_flag)
    ret = OBJ(del);
  else
    ret = (void*)HBT_DELETED;

  _free_HBT_element(tree, del);

  tree->num--;

  return ret;
}

/*---------------------------------------------------------------------------*/
/* Deletes an object from the tree.                                          */
/* If user is controlling allocation returns pointer to obj found.           */
/*---------------------------------------------------------------------------*/
void *HBT_successor(
                    HBT * tree,
                    void *obj)
{
  /*-----------------------------------------------------------------------*/
  /* The stack keeps elements from root to the node to be deleted.         */
  /* element_stack has pointers to the nodes.  dir_stack has the direction */
  /* taken down the path.                                                  */
  /*-----------------------------------------------------------------------*/
  int size = 0;
  HBT_element *temp;
  HBT_element *element_stack[STACK_SIZE];
  short dir_stack[STACK_SIZE];

  int (*compare)(void *, void *) = tree->compare;
  int test;


  /*-----------------------------------------------------------------------*/
  /* Find the node to be deleted.  Keep track of path on the stack.        */
  /*-----------------------------------------------------------------------*/

  if (!(temp = tree->root))
    return NULL;

  /* If obj is NULL then find the first node in the tree                   */
  if (!obj)
  {
    while (LEFT(temp))
      temp = LEFT(temp);

    return temp->obj;
  }


  while (temp && (test = (*compare)(obj, OBJ(temp))))
  {
    element_stack[++size] = temp;
    if (test < 0)
    {
      dir_stack[size] = UNBALANCED_LEFT;
      temp = LEFT(temp);
    }
    else
    {
      dir_stack[size] = UNBALANCED_RIGHT;
      temp = RIGHT(temp);
    }
  }

  if (temp == NULL)
    return NULL;

  /* If this node has right children then the successor is the first       */
  /* descendant with no left children.                                      */
  /* Otherwise the successor is the first predecessor with right children  */
  if (RIGHT(temp))
  {
    temp = RIGHT(temp);
    while (LEFT(temp))
      temp = LEFT(temp);
  }
  else
  {
    while ((dir_stack[size] == UNBALANCED_RIGHT) && size)
      size--;

    if (size)
      temp = element_stack[size];
    else
      return NULL;
  }

  return temp->obj;
}

/*===========================================================================*/
/* Prints HBT to a file.  Recursive.                                         */
/*===========================================================================*/
void _HBT_printf(
                 FILE *file,
                 void (*printf_method)(FILE *, void *),
                 HBT_element *tree)
{
  if (tree != NULL)
  {
    _HBT_printf(file, printf_method, tree->left);
    (*printf_method)(file, tree->obj);
    _HBT_printf(file, printf_method, tree->right);
  }
}

/*===========================================================================*/
/* Print the current contents of the tree.                                   */
/*===========================================================================*/
void HBT_printf(
                FILE *file,
                HBT * tree)
{
  fprintf(file, "# %d\n", tree->height);
  fprintf(file, "# %d\n", tree->num);

  /* start the recursive printing  */
  _HBT_printf(file, tree->printf, tree->root);
}

/*===========================================================================*/
/* Scan the current contents of the tree.                                   */
/*===========================================================================*/
void HBT_scanf(
               FILE *file,
               HBT * tree)
{
  int i;
  int height, num;
  void *obj;
  int size;

  if (fscanf(file, "%d", &(height)) != 1)
  {
    printf("ERROR: HBT_scanf failed to read height\n");
    abort();
  }

  if (fscanf(file, "%d", &(num)) != 1)
  {
    printf("ERROR: HBT_scanf failed to read num\n");
    abort();
  }

  i = num;
  while (i--)
  {
    size = (*tree->scanf)(file, &obj);
    HBT_insert(tree, obj, size);
  }
}





