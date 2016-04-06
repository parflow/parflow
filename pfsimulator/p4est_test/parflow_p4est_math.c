#include <math.h>

int
gcd (int a, int b)
{
  int                 c;

  while (a) {
    c = a;
    a = b % a;
    b = c;
  }
  return b;
}

int
powtwo_div (int a)
{
  int                 c = 0;

  while (!(a % 2)) {
    c++;
    a /= 2;
  }
  return (int) pow (2, c);
}
