"""Matlab(tm) compatibility functions.

This will hopefully become a complete set of the basic functions available in
matlab.  The syntax is kept as close to the matlab syntax as possible.  One 
fundamental change is that the first index in matlab varies the fastest (as in 
FORTRAN).  That means that it will usually perform reductions over columns, 
whereas with this object the most natural reductions are over rows.  It's perfectly
possible to make this work the way it does in matlab if that's desired.
"""
from Numeric import *

# Elementary Matrices

# zeros is from matrixmodule in C
# ones is from Numeric.py

import RandomArray
def rand(*args):
	"""rand(d1,...,dn) returns a matrix of the given dimensions
	which is initialized to random numbers from a uniform distribution
	in the range [0,1).
	"""
	return RandomArray.random(args)

def randn(*args):
    """u = randn(d0,d1,...,dn) returns zero-mean, unit-variance Gaussian
    random numbers in an array of size (d0,d1,...,dn)."""
    x1 = RandomArray.random(args)
    x2 = RandomArray.random(args)
    return sqrt(-2*log(x1))*cos(2*pi*x2)
 

def eye(N, M=None, k=0, typecode=None):
	"""eye(N, M=N, k=0, typecode=None) returns a N-by-M matrix where the 
	k-th diagonal is all ones, and everything else is zeros.
	"""
	if M == None: M = N
	if type(M) == type('d'): 
		typecode = M
		M = N
	m = equal(subtract.outer(arange(N), arange(M)),-k)
	return asarray(m,typecode=typecode)


def tri(N, M=None, k=0, typecode=None):
	"""tri(N, M=N, k=0, typecode=None) returns a N-by-M matrix where all
	the diagonals starting from lower left corner up to the k-th are all ones.
	"""
	if M == None: M = N
	if type(M) == type('d'): 
		typecode = M
		M = N
	m = greater_equal(subtract.outer(arange(N), arange(M)),-k)
	return m.astype(typecode)
	
# Matrix manipulation

def diag(v, k=0):
	"""diag(v,k=0) returns the k-th diagonal if v is a matrix or
	returns a matrix with v as the k-th diagonal if v is a vector.
	"""
	v = asarray(v)
	s = v.shape
	if len(s)==1:
		n = s[0]+abs(k)
		if k > 0:
			v = concatenate((zeros(k, v.typecode()),v))
		elif k < 0:
			v = concatenate((v,zeros(-k, v.typecode())))
		return eye(n, k=k)*v
	elif len(s)==2:
		v = add.reduce(eye(s[0], s[1], k=k)*v)
		if k > 0: return v[k:]
		elif k < 0: return v[:k]
		else: return v
	else:
	        raise ValueError, "Input must be 1- or 2-D."
	

def fliplr(m):
	"""fliplr(m) returns a 2-D matrix m with the rows preserved and
	columns flipped in the left/right direction.  Only works with 2-D
	arrays.
	"""
	m = asarray(m)
	if len(m.shape) != 2:
		raise ValueError, "Input must be 2-D."
	return m[:, ::-1]

def flipud(m):
	"""flipud(m) returns a 2-D matrix with the columns preserved and
	rows flipped in the up/down direction.  Only works with 2-D arrays.
	"""
	m = asarray(m)
	if len(m.shape) != 2:
		raise ValueError, "Input must be 2-D."
	return m[::-1]
	
# reshape(x, m, n) is not used, instead use reshape(x, (m, n))

def rot90(m, k=1):
	"""rot90(m,k=1) returns the matrix found by rotating m by k*90 degrees
	in the counterclockwise direction.
	"""
	m = asarray(m)
	if len(m.shape) != 2:
		raise ValueError, "Input must be 2-D."
	k = k % 4
	if k == 0: return m
	elif k == 1: return transpose(fliplr(m))
	elif k == 2: return fliplr(flipud(m))
	elif k == 3: return fliplr(transpose(m))

def tril(m, k=0):
	"""tril(m,k=0) returns the elements on and below the k-th diagonal of
	m.  k=0 is the main diagonal, k > 0 is above and k < 0 is below the main
	diagonal.
	"""
	return tri(m.shape[0], m.shape[1], k=k, typecode=m.typecode())*m

def triu(m, k=0):
	"""triu(m,k=0) returns the elements on and above the k-th diagonal of
	m.  k=0 is the main diagonal, k > 0 is above and k < 0 is below the main
	diagonal.
	"""	
	return (1-tri(m.shape[0], m.shape[1], k-1, m.typecode()))*m 

# Data analysis

# Basic operations
def max(m,axis=0):
	"""max(m,axis=0) returns the maximum of m along dimension axis.
	"""
	return maximum.reduce(m,axis)

def min(m,axis=0):
	"""min(m,axis=0) returns the minimum of m along dimension axis.
	"""
	return minimum.reduce(m,axis)

# Actually from BASIS, but it fits in so naturally here...

def ptp(m,axis=0):
	"""ptp(m,axis=0) returns the maximum - minimum along the the given dimension
	"""
	return max(m,axis)-min(m,axis)

def mean(m,axis=0):
	"""mean(m,axis=0) returns the mean of m along the given dimension.
	Note:  if m is an integer array, integer division will occur.
	"""
	return add.reduce(m,axis)/m.shape[axis]

# sort is done in C but is done row-wise rather than column-wise
def msort(m):
	"""msort(m) returns a sort along the first dimension of m as in MATLAB.
	"""
	return transpose(sort(transpose(m)))

def median(m):
	"""median(m) returns a mean of m along the first dimension of m.
	"""
	return msort(m)[m.shape[0]/2.0]

def std(m,axis=0):
	"""std(m,axis=0) returns the standard deviation along the given 
	dimension of m.  The result is unbiased with division by N-1.
	"""
	mu = mean(m,axis)
	return sqrt(add.reduce(pow(m-mu,2),axis))/sqrt(m.shape[axis]-1.0)

def sum(m,axis=0):
	"""sum(m,axis=0) returns the sum of the elements of m along the given dimension.
	"""
	return add.reduce(m,axis)

def cumsum(m,axis=0):
	"""cumsum(m,axis=0) returns the cumulative sum of the elements along the
	given dimension of m.
	"""
	return add.accumulate(m,axis)

def prod(m,axis=0):
	"""prod(m,axis=0) returns the product of the elements along the given
	dimension of m.
	"""
	return multiply.reduce(m,axis)

def cumprod(m,axis=0):
	"""cumprod(m) returns the cumulative product of the elments along the
	given dimension of m.
	"""
	return multiply.accumulate(m,axis)

def trapz(y, x=None, axis=-1):
	"""trapz(y,x=None,axis) integrates y along the given dimension of
	the data array using the trapezoidal rule.
	"""
	if x == None:
		d = 1.0
	else:
		d = diff(x,axis=axis)
	y = asarray(y)
	nd = len(y.shape)
	slice1 = [slice(None)]*nd
	slice2 = [slice(None)]*nd
	slice1[axis] = slice(1,None)
	slice2[axis] = slice(None,-1)
	return add.reduce(d * (y[slice1]+y[slice2])/2.0,axis)

def diff(x, n=1,axis=-1):
	"""diff(x,n=1) calculates the first-order, discrete difference
	approximation to the derivative along the axis specified.
	"""
	x = asarray(x)
	nd = len(x.shape)
	slice1 = [slice(None)]*nd
	slice2 = [slice(None)]*nd
	slice1[axis] = slice(1,None)
	slice2[axis] = slice(None,-1)
	if n > 1:
		return diff(x[slice1]-x[slice2], n-1)
	else:
		return x[slice1]-x[slice2]

def corrcoef(x, y=None):
	"""The correlation coefficients
	"""
	c = cov(x, y)
	d = diag(c)
	return c/sqrt(multiply.outer(d,d))

def cov(m,y=None):
	m = asarray(m)
	mu = mean(m)
	if y != None: m = concatenate((m,y))
	sum_cov = 0.0
	for v in m:
		sum_cov = sum_cov+multiply.outer(v,v)
	return (sum_cov-len(m)*multiply.outer(mu,mu))/(len(m)-1.0)

# Added functions supplied by Travis Oliphant
import LinearAlgebra
def squeeze(a):
    "squeeze(a) returns a with any ones from the shape of a removed"
    b = asarray(a.shape)
    return reshape (a, tuple (compress (not_equal (b, 1), b)))


def kaiser(M,beta):
    """kaiser(M, beta) returns a Kaiser window of length M with shape parameter
    beta. It depends on the cephes module for the modified bessel function i0.
    """
    import cephes
    n = arange(0,M)
    alpha = (M-1)/2.0
    return cephes.i0(beta * sqrt(1-((n-alpha)/alpha)**2.0))/cephes.i0(beta)

def blackman(M):
    """blackman(M) returns the M-point Blackman window.
    """
    n = arange(0,M)
    return 0.42-0.5*cos(2.0*pi*n/M) + 0.08*cos(4.0*pi*n/M)


def bartlett(M):
    """bartlett(M) returns the M-point Bartlett window.
    """
    n = arange(0,M)
    return where(less_equal(n,M/2.0),2.0*n/M,2.0-2.0*n/M)

def hanning(M):
    """hanning(M) returns the M-point Hanning window.
    """
    n = arange(0,M)
    return 0.5-0.5*cos(2.0*pi*n/M)

def hamming(M):
    """hamming(M) returns the M-point Hamming window.
    """
    n = arange(0,M)
    return 0.54-0.46*cos(2.0*pi*n/M)

def sinc(x):
    """sinc(x) returns sin(pi*x)/(pi*x) at all points of array x.
    """
    return where(equal(x,0.0),1.0,sin(pi*x)/(pi*x))


def eig(v):
    """[x,v] = eig(m) returns the eigenvalues of m in x and the corresponding
    eigenvectors in the rows of v.
    """
    return LinearAlgebra.eigenvectors(v)

def svd(v):
    """[u,x,v] = svd(m) return the singular value decomposition of m.
    """
    return LinearAlgebra.singular_value_decomposition(v)


def angle(z):
    """phi = angle(z) return the angle of complex argument z."""
    z = asarray(z)
    if z.typecode() in ['D','F']:
       zimag = z.imag
       zreal = z.real
    else:
       zimag = 0
       zreal = z
    return arctan2(zimag,zreal)


def roots(p):
	"""r = roots(p)

	return the roots of the polynomial whose coefficients are the elements
	of the rank-1 array p.  If the length of p is n+1 then 
	the polynomial is p[0]*x**n + ... + p[n-1]*x + p[n]
	"""
	p = 1.0*asarray(p)
	n = len(p)
	if len(p.shape) != 1:
		raise ValueError, "Input must be a rank-1 array."

	# Strip zeros at front of array
	ind = 0
	while (p[ind] == 0):
		ind = ind + 1
	p = p[ind:]

	N = len(p)
	root = zeros((N-1,),'D')
	# Strip zeros at end of array which correspond to zero-valued roots
	ind = len(p)
	while (p[ind-1]==0):
		ind = ind - 1
	p = p[:ind]

	N = len(p)
	if N > 1:
		A = diag(ones((N-2,),p.typecode()),-1)
		A[0,:] = -p[1:] / p[0]
		root[:N-1] = eig(A)[0]
	return root


























