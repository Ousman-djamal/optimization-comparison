import numpy as np
import matplotlib.pyplot as plt

# --- Calcul du gradient de la fonction J ---
def grad_J(x, y):

    df_dx = 2*(x - 1) + 40*x*(x**2 - y)
    df_dy = -20*(x**2 - y)

    return np.array([df_dx, df_dy])

# --- Algorithme du gradient à pas constant ---
def gradient_descent(start_point, alpha,
                     n_iterations=1000,
                     tol=1e-6):

    # Conversion en tableau numpy
    x_curr = np.array(start_point, dtype=float)

    # Historique des points
    # utile pour afficher la trajectoire
    history = [x_curr.copy()]

    for i in range(n_iterations):

        # Calcul du gradient au point courant
        gradient = grad_J(x_curr[0], x_curr[1])

        # Mise à jour :
        # X(k+1) = X(k) - alpha * grad(J)
        x_next = x_curr - alpha * gradient

        # Vérification de la convergence
        if np.linalg.norm(x_next - x_curr) < tol:
            break

        # Mise à jour du point courant
        x_curr = x_next

        # Sauvegarde du point
        history.append(x_curr.copy())

    return x_curr, np.array(history)

# --- Paramètres ---
point_initial = [-1, 1]

# Choix du pas constant
pas = 0.01

# --- Exécution ---
resultat, trajectoire = gradient_descent(point_initial, pas)

print(f"Minimum trouvé en : {resultat}")
print(f"Nombre d'itérations : {len(trajectoire)}")

# ---------------------------------------------------
#                PARTIE VISUALISATION
# -----------------------------------------------------

# Création de la grille
x_range = np.linspace(-1.5, 1.5, 100)
y_range = np.linspace(-0.5, 2.0, 100)

X_mesh, Y_mesh = np.meshgrid(x_range, y_range)

# Calcul de la fonction sur toute la grille
Z_mesh = (X_mesh - 1)**2 + 10 * (X_mesh**2 - Y_mesh)**2

# Création de la figure
plt.figure(figsize=(10, 8))

# Courbes de niveau
levels = np.logspace(-1, 2, 15)

plt.contourf(X_mesh,
             Y_mesh,
             Z_mesh,
             levels=levels,
             cmap='viridis',
             alpha=0.6)

# Barre de couleurs
plt.colorbar(label='J(x,y)')

# --- Affichage de la trajectoire ---
plt.plot(trajectoire[:, 0],
         trajectoire[:, 1],
         'r-o',
         markersize=3,
         label='Gradient à pas constant')

# Point initial
plt.plot(point_initial[0],
         point_initial[1],
         'go',
         markersize=8,
         label='Point initial')

# Minimum théorique
plt.plot(1, 1,
         'ro',
         markersize=8,
         label='Minimum théorique')

# Titre et axes
plt.title("Trajectoire du Gradient à Pas Constant")
plt.xlabel("x")
plt.ylabel("y")

# Affichage de la légende
plt.legend()

# Affichage de la grille
plt.grid(True, linestyle='--', alpha=0.5)

# Affichage final
plt.show()