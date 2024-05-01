import numpy as np

row_a = 10
col_b = 15
mutual = 3

a = np.arange( row_a * mutual).reshape(row_a,mutual)
b = np.arange( mutual * col_b ).reshape(mutual,col_b)
c = np.arange( mutual * col_b ).reshape(mutual,col_b).T

print("a" , a )
print("b" , b )
print("c" , c )

print( "test1" , a.dot(b) )
print( "test2" , a.dot(c.T) )
