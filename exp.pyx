from cython cimport cdivision, boundscheck

from fwht cimport _fwht_

cdef extern from "complex.h":
    double complex cexp(double complex)

@cdivision
@boundscheck(False)
cdef void _zbasis_exp_(double[:] Hp, double complex[:] psi, double t):
  cdef long lena = psi.shape[0]
  cdef long h=1
  cdef long i,j,k
  cdef double x, y

  for i in range(lena):
    psi[i] = cexp(-1.0j*t*Hp[i])*psi[i]

@cdivision
@boundscheck(False)
cdef void _xbasis_exp_(double[:] Hp, double complex[:] psi, double t):
  cdef long lena = psi.shape[0]
  cdef long h=1
  cdef long i,j,k
  cdef double x, y

  _fwht_(psi)
  for i in range(lena):
    psi[i] = cexp(-1.0j*t*Hp[i])*psi[i]
  _fwht_(psi)

cpdef void zbasis_exp(double[:] Hp, double complex[:] psi, double t):
  """
  In-place transformation of psi -> exp(-j*Hp*t)psi where Hp is a diagonal
  matrix given as a float 1-D NumPy array of shape (2^n,) with integer n, psi
  is a vector given as a complex 1-D NumPy array of shape (2^n,) and t is a
  float
  """
  _zbasis_exp_(Hp, psi, t)

cpdef void xbasis_exp(double[:] Hp, double complex[:] psi, double t):
  """
  In-place transformation of psi -> exp(-j*Hp*t)psi where Hp is a matrix that
  would be diagonalized by a Walsh-Hadamard transformation, given as the
  diagonal in that basis as a float 1-D NumPy array of shape (2^n,) with
  integer n, psi is a vector given as a complex 1-D NumPy array of shape (2^n,)
  and t is afloat
  """
  _xbasis_exp_(Hp, psi, t)
