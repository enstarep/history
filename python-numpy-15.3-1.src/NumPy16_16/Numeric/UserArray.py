from Numeric import *
import string

class UserArray:
    def __init__(self, data, typecode = None):
        self.array = array(data, typecode)
        self.shape = self.array.shape
        self._typecode = self.array.typecode()
        self.name = string.split(str(self.__class__))[0]

    def __repr__(self):
        return self.name+repr(self.array)[len("array"):]

    def __array__(self,t=None):
        if t: return asarray(self.array,t)
        return asarray(self.array)

    def __float__(self):
        return float(asarray(self.array))

    # Array as sequence
    def __len__(self): return len(self.array)

    def __getitem__(self, index): 
        return self._rc(self.array[index])

    def __getslice__(self, i, j): 
        return self._rc(self.array[i:j])


    def __setitem__(self, index, value): self.array[index] = asarray(value,self._typecode)
    def __setslice__(self, i, j, value): self.array[i:j] = asarray(value,self._typecode)

    def __del__(self):
        # necessary?
        for att in self.__dict__.keys():
            delattr(self,att)

    def __abs__(self): return self._rc(absolute(self.array))
    def __neg__(self): return self._rc(-self.array)

    def __add__(self, other): 
        return self._rc(self.array+asarray(other))
    __radd__ = __add__

    def __sub__(self, other): 
        return self._rc(self.array-asarray(other))
    def __rsub__(self, other): 
        return self._rc(asarray(other)-self.array)

    def __mul__(self, other): 
        return self._rc(multiply(self.array,asarray(other)))
    __rmul__ = __mul__

    def __div__(self, other): 
        return self._rc(divide(self.array,asarray(other)))
    def __rdiv__(self, other): 
        return self._rc(divide(asarray(other),self.array))

    def __pow__(self,other): 
        return self._rc(power(self.array,asarray(other)))
    def __rpow__(self,other): 
        return self._rc(power(asarray(other),self.array))

    def __sqrt__(self): 
        return self._rc(sqrt(self.array))

    def tostring(self): return self.array.tostring()

    def byteswapped(self): return self._rc(self.array.byteswapped())

    def astype(self, typecode): return self._rc(self.array.asType(typecode))
   
    def typecode(self): return self._typecode

    def itemsize(self): return self.array.itemsize()

    def iscontiguous(self): return self.array.iscontiguous()

    def _rc(self, a):
        if len(shape(a)) == 0: return a
        else: return self.__class__(a)

    def __setattr__(self,attr,value):
        if attr=='shape':
            self.array.shape=value
        self.__dict__[attr]=value

    def __getattr__(self,attr):
        # for .attributes for example, and any future attributes
        return getattr(self.array, attr)
            
#############################################################
# Test of class UserArray
#############################################################
if __name__ == '__main__':
    import Numeric

    temp=reshape(arange(10000),(100,100))

    ua=UserArray(temp)
    # new object created begin test
    print dir(ua)
    print shape(ua),ua.shape # I have changed Numeric.py

    ua_small=ua[:3,:5]
    print ua_small
    ua_small[0,0]=10  # this did not change ua[0,0], wich is not normal behavior
    print ua_small[0,0],ua[0,0]
    print sin(ua_small)/3.*6.+sqrt(ua_small**2)
    print less(ua_small,103),type(less(ua_small,103))
    print type(ua_small*reshape(arange(15),shape(ua_small)))
    print reshape(ua_small,(5,3))
    print transpose(ua_small)
