# -*- coding: utf-8 -*-
"""Lineal Algebra Practice 1.ipynb

# Question 1.

* Print the sum of matrixes A and B

$$ A =
\left(\begin{array}{cc} 
1 & 2\\
3 & 4\\
5 & 6
\end{array}\right) + B =
\left(\begin{array}{cc} 
6 & 5\\
4 & 3\\
2 & 1 
\end{array}\right)
$$
"""

# Answer 1.
'Your code here'

import numpy as np
A = np.array([[1, 2], [3, 4], [5, 6]])
B = np.array([[6, 5], [4, 3], [2, 1]])
print(A+B)

"""# Question 2

* Print the sum of matrix + scalar.

$$ M =
\left(\begin{array}{cc} 
6 & 3\\
9 & 5\\
4 & 8
\end{array}\right) + N =
\left(\begin{array}{cc} 
2.5 \\ 
\end{array}\right)
$$
"""

# Answer 2.
'Your code here'
M = np.array([[6, 3], [9, 5], [4, 8]])
N = 2.5
print(M+N)

"""# Question 3.

* Print the transposed matrix of  $J$  -->  $J^t$. 

$$ J =
\left(\begin{array}{cc} 
9 & 2 & 3\\
3 & 7 & 4 \\
5 & 6 & 9
\end{array}\right) 
$$
"""

# Answer 3.
'Your code here'
J = np.array([[9, 2, 3], [3, 7, 4], [5, 6, 9]])
print(J.T)

"""# Question 4.

* For the $ J ^ t $ matrix, calculate your transpose to obtain the original matrix J. 

$$ J =
\left(\begin{array}{cc} 
9 & 2 & 3\\
3 & 7 & 4 \\
5 & 6 & 9
\end{array}\right) 
$$
"""

# Answer 4.
'Your code here'
print((J.T).T)

"""# Question 5.

* 
What are the missing values? 

$$
\left(\begin{array}{cc} 
? & 2\\
0 & 2
\end{array}\right)
\left(\begin{array}{cc} 
1 & 0 \\ 
1 & ? 
\end{array}\right)
$$ 

 * so that the answer is like this:
$
\left(
 \begin {array}{ll} 
1 & 0 \\
0   & 0
\end{array}
\right)
$

- [ A ] = 1 y 1
- [ B ] = 0 y 1
- [ C ] = 1 y 0
- [ D ] = 0 y 0

*  Print the result and the answer.
"""

# Answer 5.
'your code here'
A = np.array([[1, 2], [0, 2]])
D = np.array([[0, 2], [0, 2]])


C = A * d
print(C)
"""# Question 6.
* ¿ What is the dot product matrixes  $L$ y $K$ = : ?

$$ L =
\left(\begin{array}{cc} 
3 & 5 \\
7 & 1 \\
\end{array}\right) K =
\left(\begin{array}{cc} 
3 & 12  \\
6 & 4  \\
\end{array}\right)
$$
"""

# Answer 6.
'your code here'
L = np.array([[3, 5], [7, 1]])
K = np.array([[3, 12], [6, 4]])
print(L.dot(K))

"""# Question 7.

* ¿ What is the determinant of the following matrix  ? (Print the answer)

$
\left(
\begin {array}{ll} 
1 & 2 & 0 \\
1 & 2 & 0 \\
0 & 1 & 1 \\
\end{array}
\right)
$

- [ A ] = 1 
- [ B ] = 0 
- [ C ] = 5
- [ D ] = -5
"""

# Answer 7.
'your code here'
a = np.array([[1, 2, 0], [1, 2, 0], [0, 1, 1]])
print(np.linalg.det(a))

"""# Question 8.

* ¿ What is the determinant of the following matrix ? (Print the answer)

$
\left(
\begin {array}{ll} 
0 & 1 & 1 \\
1 & 0 & 0 \\
0 & 0 & 1 \\
\end{array}
\right)
$

- [ A ] = 0 
- [ B ] = -1 
- [ C ] = 3
- [ D ] = 1
"""

# Answer 8.
'your code here'
b = np.array([[0, 1, 1], [1, 0, 0], [0, 0, 1]])
print(np.linalg.det(b))

"""# Question 9

* ¿What is the missing value?
$$ 
\left(\begin{array}{cc} 
1 & 2 & 3\\
4 & 5 & 6\\
7 & 8 & 9
\end{array}\right) 
\left(\begin{array}{cc} 
0 \\ 
? \\
16 \\
\end{array}\right) =
\left(\begin{array}{cc} 
4 \\ 
10 \\
16 \\
\end{array}\right)
$$
"""

matrix_1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
matrix_2 = np.array ([0, 2, 0])
answer = matrix_1 *matrix_2

print(answer)
