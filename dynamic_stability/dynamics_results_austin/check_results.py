import numpy as np
from scipy.linalg import eig
 

# my method
A_matrix1 = np.load('A_matrix_m1.npy')
B_matrix1 = np.load('B_matrix_m1.npy')

# my simpler method
A_matrix2 = np.load('A_matrix_m2.npy')
B_matrix2 = np.load('B_matrix_m2.npy')

# austins method
A_matrix_austin = np.load('A_matrix_maustin.npy')
B_matrix_austin = np.load('B_matrix_maustin.npy')



# diff_A = 100*(A_matrix - A)/A
diff_A1 = (A_matrix1 - A_matrix2)

diff_B1 = (B_matrix1 - B_matrix2)


diff_A2 = (A_matrix1 - A_matrix_austin)

diff_B2 = (B_matrix1 - B_matrix_austin)

# C_matrix = np.matmul(np.linalg.inv(B),A)
# eigvals, eigvecs = eig(C_matrix)

# print('\n')
# print('C-Matrix')
# print('\n'.join([''.join(['{:>12.6f}'.format(item) for item in row]) 
#         for row in C_matrix]))

# #normalize eigenvectors relative to the largest in each array
# for i in range(12):
#     index_max = np.argmax(np.abs(eigvecs[:,i]))
    
#     cc = np.conj(eigvecs[index_max,i])
    
#     new_vec = cc*eigvecs[:,i]
    
#     new_vec = new_vec / np.sqrt(np.sum(np.square(np.abs(new_vec))))
    
#     eigvecs[:,i] = new_vec

# i_sort = np.argsort(np.abs(eigvals))

# eigvals = eigvals[i_sort]

# eigvecs = eigvecs[:,i_sort]

# print('\nEigenvalues') 
# print('\n'.join('{:>32.12f}'.format(item) for item in eigvals))

# print('\nEigenvectors (1-6)') 
# print('\n'.join([''.join(['{:>26.8f}'.format(item) for item in row]) 
#         for row in eigvecs[:,:6]]))

# print('\nEigenvectors (7-12)') 
# print('\n'.join([''.join(['{:>26.8f}'.format(item) for item in row]) 
#         for row in eigvecs[:,6:]]))