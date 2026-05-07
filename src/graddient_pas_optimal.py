import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# --- ÉTAPE 1 : Définition de la fonction et du gradient ---
def J(X):
    x, y = X[0], X[1]
    return (x - 1)**2 + 10 * (x**2 - y)**2

def grad_J(X):
    x, y = X[0], X[1]
    df_dx = 2*(x - 1) + 40*x*(x**2 - y)
    df_dy = -20*(x**2 - y)
    return np.array([df_dx, df_dy])

# --- ÉTAPE 2 : Algorithme du gradient à pas optimal ---
def gradient_pas_optimal(X0, n_iterations=100, tol=1e-5):
    X_curr = np.array(X0, dtype=float)
    history = [X_curr.copy()]
    
    for i in range(n_iterations):
        g = grad_J(X_curr)
        
        # Si le gradient est presque nul, on a convergé
        if np.linalg.norm(g) < tol:
            break
            
        # Définition de la fonction de recherche de pas : phi(alpha) = J(X - alpha * g)
        def phi(alpha):
            return J(X_curr - alpha * g)
        
        # Recherche numérique du pas optimal alpha
        # On cherche alpha entre 0 et un maximum raisonnable
        res_step = minimize_scalar(phi, bounds=(0, 1), method='bounded')
        alpha_opt = res_step.x
        
        # Mise à jour du point
        X_next = X_curr - alpha_opt * g
        
        # Critère d'arrêt sur le déplacement
        if np.linalg.norm(X_next - X_curr) < tol:
            break
            
        X_curr = X_next
        history.append(X_curr.copy())
        
    return X_curr, np.array(history)

# --- ÉTAPE 3 : Exécution ---
X0 = [-1.0, 1.0]
point_final, trajectoire = gradient_pas_optimal(X0)

print(f"Minimum atteint en : {point_final}")
print(f"Nombre d'itérations : {len(trajectoire)}")
# --- ÉTAPE 4 : Visualisation de la trajectoire ---
x_range = np.linspace(-1.5, 1.5, 100)
y_range = np.linspace(-0.5, 2.0, 100)
X_mesh, Y_mesh = np.meshgrid(x_range, y_range)
Z_mesh = (X_mesh - 1)**2 + 10 * (X_mesh**2 - Y_mesh)**2

plt.figure(figsize=(10, 8))

# Dessin des courbes de niveau (échelle logarithmique pour mieux voir la vallée)
levels = np.logspace(-1, 2, 15)
plt.contourf(X_mesh, Y_mesh, Z_mesh, levels=levels, cmap='viridis', alpha=0.6)
plt.colorbar(label='J(x, y)')

# Dessin de la trajectoire
plt.plot(trajectoire[:, 0], trajectoire[:, 1], 'r-o', markersize=4, label='Chemin du gradient')
plt.plot(X0[0], X0[1], 'go', markersize=8, label='Départ (-1, 1)')
plt.plot(1, 1, 'ro', markersize=8, label='Minimum (1, 1)')

plt.title('Trajectoire du Gradient à Pas Optimal')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.show()