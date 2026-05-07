import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# --- ÉTAPE 1 : Définition mathématique ---
def J(x, y):
    # Cette fonction implémente J(x, y) = (x - 1)² + 10(x² - y)²
    # x et y peuvent être des nombres ou des matrices (tableaux NumPy)
    return (x - 1)**2 + 10 * (x**2 - y)**2

# --- ÉTAPE 2 : Préparation de la "grille" de calcul ---
# linspace crée 100 points régulièrement espacés entre les bornes
x_range = np.linspace(-2, 2, 100) 
y_range = np.linspace(-1, 3, 100)

# meshgrid transforme ces deux vecteurs en matrices X et Y.
# Cela crée une grille de coordonnées (x,y) pour chaque point du plan.
X, Y = np.meshgrid(x_range, y_range)

# On calcule la valeur de J (la hauteur Z) pour chaque point (x,y) de la grille
Z = J(X, Y)

# --- ÉTAPE 3 : Création de la figure ---
fig = plt.figure(figsize=(12, 8))
# On précise projection='3d' pour pouvoir manipuler l'axe de profondeur (Z)
ax = fig.add_subplot(111, projection='3d')

# --- ÉTAPE 4 : Dessin de la surface ---
# cmap=cm.viridis : définit le dégradé de couleur (du bleu au jaune)
# antialiased=True : lisse les traits pour un rendu plus propre
surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, antialiased=True, alpha=0.8)

# Ajout d'un marqueur pour le minimum global calculé précédemment (1, 1, 0)
# s=100 définit la taille du point
ax.scatter(1, 1, 0, color='red', s=100, label='Minimum global (1, 1, 0)', zorder=5)

# --- ÉTAPE 5 : Habillage et Labels ---
ax.set_title("Topographie de la fonction J (Surface 3D)")
ax.set_xlabel('Axe X')
ax.set_ylabel('Axe Y')
ax.set_zlabel('Valeur de J(x, y)')

# Ajout de la barre de légende pour l'échelle des couleurs
fig.colorbar(surf, shrink=0.5, aspect=10)
ax.legend()

# view_init : oriente la caméra (élévation de 30°, rotation de 150°)
ax.view_init(elev=30, azim=150)

plt.show()