cimport numpy as np
import numpy as np

from cython cimport cdivision, boundscheck

from exp cimport _xbasis_exp_,_zbasis_exp_

cdef extern from "complex.h":
    double complex conj(double complex)
    double creal(double complex)

@cdivision
@boundscheck(False)
cdef void _aqc_(double[:] Hz, double[:] Hx, double[:] s, double complex[:] psi, double Tmax, long sol, double[:] probs):
  cdef long len_s = s.shape[0]
  cdef long i
  cdef double dt = Tmax/len_s
  cdef double sval, sval_com

  for i in range(len_s):
    probs[i] = creal(psi[sol]*conj(psi[sol]))
    sval = s[i]
    sval_com = 1.0 - s[i]
    _zbasis_exp_(Hz, psi, sval*dt)
    _xbasis_exp_(Hx, psi, sval_com*dt)
  probs[i+1] = creal(psi[sol]*conj(psi[sol]))

cpdef np.ndarray[np.double_t, ndim=1] aqc(double[:] Hz, double[:] Hx, \
double[:] s, double complex[:] psi, double Tmax, long sol):
  """
  Performs a QAOA simulation on spins where the driver is diagonal in the
  X-basis and the problem is diagonal in the Z-basis
  === Inputs ===
  Hz:   The problem Hamiltonian, given as a 1-D float NumPy array of shape
        (2^n,), where n is the number of spins, containing diag(Hz)

  Hx:   The driver Hamiltonian, given as a 1-D float NumPy array of shape
        (2^n,), where n is the number of spins, containing diag(H*Hx*H) where
        H is a Hadamard on all spins

  s:    A 1-D float NumPy array containing the s(t) values in the Hamiltonian
        [1-s(t)]Hx + s(t)Hp. This assumes equally-spaced time points and
        implicitly defines the number of time-steps

  psi:  The initial state, given as a 1-D complex NumPy array of shape (2^n,).
        This will be modified in place rather than returned

  Tmax: The total run-time, given as a float

  sol:  The index representing the solution state in Z-basis

  === Returns ===
  probs: A NumPy array of shape (s.shape[0],) containing the success-probability
         if measured at that step
  """
  cdef np.ndarray[np.double_t, ndim=1] probs = \
  np.empty(s.shape[0], dtype=np.double)
  _aqc_(Hz, Hx, s, psi, Tmax, sol, probs)
  return probs
