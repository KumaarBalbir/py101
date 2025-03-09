# numpy is something which is foundation for other libraries in python such as, tf, pandas, scipy, matplotlib, scikit-learn, etc
# what is numpy?
# numpy is a fundamental package for scientific computing in python. It provides classes and functions for working with arrays, and a large collection of mathematical functions to operate on these arrays


import numpy as np

# 1D array from list
py_list = [1, 2, 3, 4, 5]
array = np.array(py_list)

type(array)  # <class 'numpy.ndarray'>

# 2D array from list
py_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
array = np.array(py_list)

type(array)  # <class 'numpy.ndarray'>


# creating arrays from scratch

np.zeros((2, 3))  # 2 rows and 3 columns fill with 0.
# 2 rows and 4 columns fill with random numbers between 0 and 1
np.random.random((2, 4))

np.arange(-3, 4)  # create an array from -3 to 4 with step size of 1
np.arange(4)  # create an array from 0 to 4 with step size of 1
np.arange(-3, 4, 2)  # create an array from -3 to 4 with step size of 2

# 3D arrays
arr_1_2D = np.array([[1, 2, 3], [4, 5, 6]])
arr_2_2D = np.array([[7, 8, 9], [10, 11, 12]])
arr_3_2D = np.array([[13, 14, 15], [16, 17, 18]])

arr_3D = np.array([arr_1_2D, arr_2_2D, arr_3_2D])

# Vector arrays
shape(5,)  # 1D array as row vector or column vector both can is true
# but shape(5,1) is not equal to shape(1,5) since shape(5,1) is 5 rows and 1 column and shape(1,5) is 1 row and 5 columns

# Matrix and tensor arrays
# a matrix has 2 dimensions and a tensor has 3 or more dimensions

# Shape Shifting

# array attribute : array.shape
# array methods: array.reshape(), array.flatten()

arr1 = np.zeros((3, 5))
print(arr1.shape)  # (3, 5)

# Flattening an array
arr2 = np.array([[1, 2], [5, 7], [6, 6]])
print(arr2.flatten())  # array([1, 2, 5, 7, 6, 6])

# Reshaping an array
arr3 = np.array([[1, 2], [5, 7], [6, 6]])
print(arr3.reshape(2, 3))  # array([[1, 2, 5], [7, 6, 6]])

# what if,
# ValueError: cannot reshape array of size 6 into shape (3, 3)
arr3.reshape((3, 3))

# sample python data types

# int, float, complex, bool, str, list, tuple, set, dict

# Numpy data types

# np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64
# np.float16, np.float32, np.float64, np.complex64, np.complex128
# np.bool, np.str, np.unicode, np.datetime64, np.timedelta64

# The .dtype attribute
print(np.array([1.32, 5.6, 2.2]).dtype)  # float64

print(np.array([1, 5, 2]).dtype)  # int64

print(np.array(["intro", "to", "numpy"]).dtype)  # dtype('<U12')

# dtype as an argument
arr = np.array([1, 5, 2], dtype=np.int64)

print(arr.dtype)  # int64

# Type conversion
bool_arr = np.array([True, False, True], dtype=np.bool_)
bool_arr.astype(np.int64)  # array([1, 0, 1], dtype=int64)

# Type Coercion
# array(['True', 'foo', '42', '5.4'], dtype='<U5')
np.array([True, "foo", 42, 5.4])

# Type Coersion hierarchy:
# adding a flot to an array of integers will convert the integers to float
np.array([0, 42, 4.5]).dtype  # float64

# adding an int to an array of booleans will change the all the booleans to integers
np.array([True, False, True, 42]).dtype  # int64

# *************************** Indexing and Slicing *************************

arr = np.array([2, 4, 6, 8, 10])
arr[3]  # 8

arr[0:3]  # array([2, 4, 6])

arr[-1]  # 10

# indexing in 2D arrays
arr_2D = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr_2D[0, 1]  # 2

arr_2D[0]  # array([1, 2, 3])

arr_2D[:, 2]  # array([3, 6, 9])

arr_2D[0:2, 1]  # array([2, 5])

arr_2D[1:3, 0:2]  # array([[4, 5], [7, 8]])

# slicing with steps
arr_2D[0:3:2, 1:3:2]  # array([[2], [8]]) # step = 2

# sorting arrays
arr2d = np.array([[3, 2], [5, 1]])
np.sort(arr2d)  # array([[2, 3], [1, 5]]) # row by row sorting

# sorting along an axis
# axis 0: along column, axis 1: along row

np.sort(arr2d, axis=0)  # array([[3, 1], [5, 2]]) # column by column sorting

np.sort(arr2d, axis=1)  # array([[2, 3], [1, 5]]) # row by row sorting

# Filtering Arrays
# Two ways: 1. Masks and fancy indexing 2. np.where()

one_to_five = np.arange(1, 6)  # array([1, 2, 3, 4, 5])
mask = one_to_five % 2 == 0  # array([False, True, False, True, False])

one_to_five[mask]  # array([2, 4])

one_to_five[one_to_five % 2 == 0]  # array([2, 4])
