
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Creazione dei dati
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
x, y = np.meshgrid(x, y)
z = np.log(np.sqrt(x**2 + y**2) + 1)

# Creazione del grafico 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Superficie
surf = ax.plot_surface(x, y, z, cmap='jet')

# Personalizzazione dei font degli assi con i nomi corretti
ax.set_xlabel('Bending Moment [Nmm]', fontsize=14, fontweight='bold')
ax.set_ylabel('Torsional moment [Nmm]', fontsize=14, fontweight='bold')
ax.set_zlabel('Total deformation load', fontsize=14, fontweight='bold')

# Personalizzazione della rotazione dei tick labels
ax.tick_params(axis='x', labelsize=10, rotation=45)
ax.tick_params(axis='y', labelsize=10, rotation=-45)
ax.tick_params(axis='z', labelsize=10)

# Aggiunta della barra di colore
fig.colorbar(surf)

# Mostra il grafico
plt.show()