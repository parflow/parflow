#include <math.h>
#include <stdint.h>

int
parflow_p4est_gcd(int a, int b)
{
  int c;

  while (a)
  {
    c = a;
    a = b % a;
    b = c;
  }
  return b;
}

int
parflow_p4est_powtwo_div(int a)
{
  int c = 0;

  while (!(a % 2))
  {
    c++;
    a /= 2;
  }
  return c;
}

int parflow_p4est_int_compare(int64_t a, int64_t b)
{
  return a == b ? 0 : a < b ? -1 : +1;
}
