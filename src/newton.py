import numpy as np
import matplotlib.pyplot as plt

# --- Définition de la fonction objectif ---
def J(X):
    x, y = X[0], X[1]
    return (x - 1)**2 + 10 * (x**2 - y)**2

# --- Calcul du gradient ---
def grad_J(X):
    x, y = X[0], X[1]

    df_dx = 2*(x - 1) + 40*x*(x**2 - y)
    df_dy = -20*(x**2 - y)

    return np.array([df_dx, df_dy])

# --- Calcul de la matrice Hessienne ---
# La Hessienne contient les dérivées secondes
def hessienne_J(X):
    x, y = X[0], X[1]

    h11 = 2 + 120*x**2 - 40*y
    h12 = -40*x
    h21 = -40*x
    h22 = 20

    return np.array([
        [h11, h12],
        [h21, h22]
    ])

# --- Méthode de Newton ---
def methode_newton(X0, n_iterations=100, tol=1e-6):

    # Conversion en tableau numpy
    X_curr = np.array(X0, dtype=float)

    # Pour stocker la trajectoire
    history = [X_curr.copy()]

    for i in range(n_iterations):

        # Calcul du gradient
        g = grad_J(X_curr)

        # Si le gradient devient très petit
        # on considère qu'on a convergé
        if np.linalg.norm(g) < tol:
            break

        # Calcul de la Hessienne
        H = hessienne_J(X_curr)

        # Direction de Newton :
        # d = - H^(-1) * grad(J)
        d = -np.linalg.inv(H) @ g

        # Mise à jour du point
        X_next = X_curr + d

        # Critère d'arrêt
        if np.linalg.norm(X_next - X_curr) < tol:
            break

        # Mise à jour
        X_curr = X_next

        # Sauvegarde du point
        history.append(X_curr.copy())

    return X_curr, np.array(history)

# --- Exécution ---
X0 = [-1.0, 1.0]

point_final, trajectoire = methode_newton(X0)

print(f"Minimum atteint en : {point_final}")
print(f"Nombre d'itérations : {len(trajectoire)}")

# --- Visualisation ---
x_range = np.linspace(-1.5, 1.5, 100)
y_range = np.linspace(-0.5, 2.0, 100)

X_mesh, Y_mesh = np.meshgrid(x_range, y_range)

Z_mesh = (X_mesh - 1)**2 + 10 * (X_mesh**2 - Y_mesh)**2

plt.figure(figsize=(10, 8))

levels = np.logspace(-1, 2, 15)

plt.contourf(X_mesh, Y_mesh, Z_mesh,
             levels=levels,
             cmap='viridis',
             alpha=0.6)

plt.colorbar(label='J(x, y)')

# Affichage de la trajectoire
plt.plot(trajectoire[:, 0],
         trajectoire[:, 1],
         'r-o',
         markersize=4,
         label='Chemin de Newton')

plt.plot(X0[0], X0[1],
         'go',
         markersize=8,
         label='Point initial')

plt.plot(1, 1,
         'ro',
         markersize=8,
         label='Minimum théorique')

plt.title("Méthode de Newton")
plt.xlabel("x")
plt.ylabel("y")

plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.show()