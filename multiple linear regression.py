import numpy as np
import math


x1 = np.array([14.62, 15.63,14.62, 15, 14.5, 15.25, 16.12, 15.13, 15.5,15.13,15.5,16.12,15.13,15.63,15.38,14.38,15.5,14.25,14.5,14.62])
x2 = np.array([226, 220, 217.4,220, 226.5, 224.1, 220.5, 223.5, 217.6,228.5,230.2,226.5,226.6,225.6,229.7,234,230,224.3,240.5,223.7])
x3 = np.array([7, 3.375,6.375, 6, 7.625,6,3.375, 6.125, 5, 6.625, 5.75,3.75,6.125,5.375,5.875,8.875,4,8,10.87,7.375])
y = np.array([128.4,52.62,113.9,98.01,139.9,102.6,48.14,109.6,82.68,112.6,97.52,59.06,111.8,89.09,101,171.9,66.8,157.1,208.4,133.4])
log_x1 = np.log(x1)
log_x2 = np.log(x2)
log_x3 = np.log(x3)
log_y = np.log(y)
A = np.column_stack((np.ones_like(log_x1), log_x1, log_x2,log_x3))
b = log_y

# Calculate (A^T A)
ATA = np.dot(A.T, A)

# Calculate (A^T b)
ATb = np.dot(A.T, b)

# Calculate the coefficients
w = np.linalg.solve(ATA, ATb)

# Extract coefficients
#a0, a1, a2 = np.exp(w)

# Display the coefficients
print(f'a0: {math.exp(w[0])}')
print(f'a1: {w[1]}')
print(f'a2: {w[2]}')
print(f'a3: {w[3]}')
# test instance 
new_diameter = 14.5
new_slope = 220
n=5
predicted_log_flow = w[0] + w[1]*np.log(new_diameter) + w[2]*np.log(new_slope) + w[3]*np.log(n)
predicted_flow = np.exp(predicted_log_flow)

# Display the predicted flow
print(f'Predicted y: {predicted_flow}')