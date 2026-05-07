import numpy as np
import matplotlib.pyplot as plt

# --- ÉTAPE 1 : Définition de la fonction ---
def J(x, y):
    return (x - 1)**2 + 10 * (x**2 - y)**2

# --- ÉTAPE 2 : Création de la grille de points ---
# On définit l'étendue des axes X et Y
x_range = np.linspace(-2, 2, 500) 
y_range = np.linspace(-1, 3, 500)
X, Y = np.meshgrid(x_range, y_range)
Z = J(X, Y)

# --- ÉTAPE 3 : Création du graphique 2D ---
plt.figure(figsize=(10, 7))

# contourf : Trace des contours remplis (f pour 'filled')
# levels=np.logspace(...) : On utilise une échelle logarithmique pour les niveaux
#  Cela permet de voir les détails près du minimum.
levels = np.logspace(-1, 3, 20)
cp = plt.contourf(X, Y, Z, levels=levels, cmap='viridis', extend='both')

# On ajoute des lignes de contour simples par-dessus pour plus de précision
line_contours = plt.contour(X, Y, Z, levels=levels, colors='white', alpha=0.3, linewidths=0.5)

# --- ÉTAPE 4 : Marquage du minimum ---
# Le point (1, 1) est le fond de la "vallée"
plt.plot(1, 1, 'ro', markersize=10, label='Minimum global (1, 1)')

# --- ÉTAPE 5 : Habillage ---
plt.colorbar(cp, label='Valeur de J(x, y)') # Affiche l'échelle des couleurs
plt.title('Lignes de niveau de J(x, y) - Vue du dessus (2D)')
plt.xlabel('Axe X')
plt.ylabel('Axe Y')
plt.legend()
plt.grid(alpha=0.3)

plt.show()