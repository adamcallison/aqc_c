from cython cimport cdivision, boundscheck
from libc.math cimport sqrt
@cdivision
@boundscheck(False)
cdef void _fwht_(double complex[:] a):
  cdef long lena = a.shape[0]
  cdef double sqrtlena = sqrt(lena)
  cdef long h=1
  cdef long i,j,k
  cdef double complex x, y

  while h < lena:
    for i from 0 <= i < lena by 2*h:
      for j in range(i, i + h):
        x = a[j]
        y = a[j+h]
        a[j] = x + y
        a[j+h] = x - y
    h = h*2
  for i in range(lena):
    a[i] = a[i]/sqrtlena

cpdef void fwht(double complex[:] a):
  """
  Performs an in-place unitary fast Walsh-Hadamard transform on complex 1-D
  NumPy array a, which should have shape (2^n,) for integer n
  """
  _fwht_(a)
