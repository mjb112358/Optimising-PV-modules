import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# To run this script, one must have the following data for the solar panel:
#   - Direct Normal Irradiance (DNI)
#   - Diffuse Horizontal Irradiance (DHI)
#   - Power output of the solar module with a two-axis tracker
#
# The data is private and cannot be shared. However, if you would like to test,
# create a CSV file with 3 columns (x, y, z):
#   - x as the power output of the PV module
#   - y as the DNI
#   - z as the DHI
# Note : modify range to match your data, standard data average over 15 min 
# ---------------------------------------------------------------------
A = np.loadtxt(".../data_example.csv", delimiter=",")  # or your actual file

L = len(A)            # Number of rows in A
Lnew = int(np.ceil(L / 300.0))

Pmax = []
DNImax = []
DHImax = []

for j in range(0, 62):  
    start_idx = 300 * j
    end_idx = 300 * (j + 1)
    
    # Safeguard if end_idx > L
    chunk = A[start_idx : min(end_idx, L), :]
    
    Pmax.append(np.max(chunk[:, 0]))
    DNImax.append(np.max(chunk[:, 1]))
    DHImax.append(np.max(chunk[:, 2]))

# Convert lists to NumPy arrays
Pmax = np.array(Pmax)
DNImax = np.array(DNImax)
DHImax = np.array(DHImax)

# ---------------------------------------------------------------------
# 3) Plot the first figure (DNImax, DHImax)
# ---------------------------------------------------------------------
plt.figure(1)
plt.plot(DNImax, label="DNImax")
plt.plot(DHImax, label="DHImax")
plt.title("DNImax and DHImax")
plt.legend()
plt.show()

# ---------------------------------------------------------------------
# 4) Solve for DNI & DHI efficiency of the PV module
# ---------------------------------------------------------------------
X = np.vstack([DNImax, DHImax])   # shape (2, N)
Y = Pmax                          # shape (N,)
X_pinv = np.linalg.pinv(X)        # shape (N, 2) for X of (2, N)

Y_row = Y.reshape(1, -1)          # shape (1, N)
Z = Y_row @ X_pinv                # shape (1, 2)
Z = Z.flatten()                   # shape (2,)

print("Coefficients (Z):", Z)

# ---------------------------------------------------------------------
# 5) Compute Pth = Z(1)*DNImax + Z(2)*DHImax
# ---------------------------------------------------------------------
Z1, Z2 = Z[0], Z[1]
Pth = Z1 * DNImax + Z2 * DHImax

# ---------------------------------------------------------------------
# 6) Plot the second figure (Pmax vs. Pth)
# ---------------------------------------------------------------------
plt.figure(2)
plt.plot(Pmax, label="Smoothed Data (Pmax)")
plt.plot(Pth, label="Regression Model (Pth)")
plt.title("Comparison: Pmax vs. Regression Model")
plt.legend()
plt.show()
