import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# --- Fonction objectif ---
def J(X):
    x, y = X[0], X[1]
    return (x - 1)**2 + 10 * (x**2 - y)**2

# --- Gradient de la fonction ---
def grad_J(X):
    x, y = X[0], X[1]

    df_dx = 2*(x - 1) + 40*x*(x**2 - y)
    df_dy = -20*(x**2 - y)

    return np.array([df_dx, df_dy])

# --- Méthode du gradient conjugué ---
def gradient_conjugue(X0, n_iterations=100, tol=1e-6):

    # Point de départ
    X_curr = np.array(X0, dtype=float)

    # Historique des points
    history = [X_curr.copy()]

    # Calcul du gradient initial
    g_curr = grad_J(X_curr)

    # Première direction :
    # d0 = -g0
    d_curr = -g_curr

    for i in range(n_iterations):

        # Si le gradient est très petit
        # on arrête l'algorithme
        if np.linalg.norm(g_curr) < tol:
            break

        # Fonction phi(alpha)
        # utilisée pour trouver le meilleur pas
        def phi(alpha):
            return J(X_curr + alpha * d_curr)

        # Recherche du pas optimal
        res = minimize_scalar(phi,
                              bounds=(0, 1),
                              method='bounded')

        alpha = res.x

        # Mise à jour du point
        X_next = X_curr + alpha * d_curr

        # Nouveau gradient
        g_next = grad_J(X_next)

        # Calcul du coefficient beta
        beta = (np.linalg.norm(g_next)**2) / (np.linalg.norm(g_curr)**2)

        # Nouvelle direction conjuguée
        d_next = -g_next + beta * d_curr

        # Critère d'arrêt
        if np.linalg.norm(X_next - X_curr) < tol:
            break

        # Mise à jour des variables
        X_curr = X_next
        g_curr = g_next
        d_curr = d_next

        # Sauvegarde
        history.append(X_curr.copy())

    return X_curr, np.array(history)

# --- Exécution ---
X0 = [-1.0, 1.0]

point_final, trajectoire = gradient_conjugue(X0)

print(f"Minimum atteint en : {point_final}")
print(f"Nombre d'itérations : {len(trajectoire)}")

# --- Visualisation ---
x_range = np.linspace(-1.5, 1.5, 100)
y_range = np.linspace(-0.5, 2.0, 100)

X_mesh, Y_mesh = np.meshgrid(x_range, y_range)

Z_mesh = (X_mesh - 1)**2 + 10 * (X_mesh**2 - Y_mesh)**2

plt.figure(figsize=(10, 8))

levels = np.logspace(-1, 2, 15)

plt.contourf(X_mesh,
             Y_mesh,
             Z_mesh,
             levels=levels,
             cmap='viridis',
             alpha=0.6)

plt.colorbar(label='J(x,y)')

# Trajectoire
plt.plot(trajectoire[:, 0],
         trajectoire[:, 1],
         'r-o',
         markersize=4,
         label='Gradient conjugué')

plt.plot(X0[0], X0[1],
         'go',
         markersize=8,
         label='Point initial')

plt.plot(1, 1,
         'ro',
         markersize=8,
         label='Minimum théorique')

plt.title("Méthode du Gradient Conjugué")
plt.xlabel("x")
plt.ylabel("y")

plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.show()