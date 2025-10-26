import string

from UserArray import UserArray, asarray
from Numeric import matrixmultiply, identity
from LinearAlgebra import inverse

class Matrix(UserArray):
    def __init__(self, data, typecode=None):
        UserArray.__init__(self,data,typecode)
        
    def __mul__(self, other):
	return self._rc(matrixmultiply(self.array, asarray(other)))

    def __rmul__(self, other):
	return self._rc(matrixmultiply(asarray(other), self.array))

    def __pow__(self, other):
        shape = self.array.shape
        if len(shape)!=2 or shape[0]!=shape[1]:
            raise TypeError, "matrix is not square"
        if type(other) in (type(1), type(1L)):
            if other==0:
                return Matrix(identity(shape[0]))
            if other<0:
                result=Matrix(inverse(self.array))
                x=Matrix(result)
                other=-other
            else:
                result=self
                x=result
            while(other>1):
                result=result*x
                other=other-1
            return result
        else:
            raise TypeError, "exponent must be an integer"

    def __rpow__(self, other):
	raise TypeError, "x**y not implemented for matrices y"


if __name__ == '__main__':
	from Numeric import *
	m = Matrix( [[1,2,3],[11,12,13],[21,22,23]])
	print m*m
	print m.array*m.array
	print transpose(m)
	print m**-1
        m = Matrix([[1,1],[1,0]])
        print "Fibonacci numbers:",
        for i in range(10):
            mm=m**i
            print mm[0][0],
        print
        
            
            
