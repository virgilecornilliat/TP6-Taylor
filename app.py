## 1. Découverte de Streamlit
# 1.1. Première application

# pour run : écrire dans le terminal : streamlit run app.py

import streamlit as st
import autograd.numpy as np
from autograd import grad
from math import factorial
import matplotlib.pyplot as plt

st.title("Approximation de Taylor")
st.header("sous-titre")
st.write("Bienvenue dans votre outil de visualisation.")

# 1.2. Afficher une formule

st.latex(r'''P_n(x) = \sum_{k=0}^{n} \frac{f^{(k)}(0)}{k!} \, x^k''')

# 1.3. Modèle d'exécution

# valeur = st.slider("Choisissez une valeur", 0, 10)
# st.write("Vous avez choisi :", valeur)

# le script s'exécute plusieurs fois

# 4. L'application : widgets et graphique

def taylor1(X, derivatives):
    """
    X : le domaine (sous-entendu X[0] < 0 et X[-1] > 0)
    derivatives : derivatives[n] = f^(n)(0), dérivée n-ième de f en 0

    Retourne Y, l'approximation de Taylor sur X pour une fonction
    qui aurait ces dérivées en 0.
    """
    Y=np.zeros(len(X))
    for i,d in enumerate(derivatives):
        Y=Y+d*X**i/factorial(i)
    return Y

def taylor2(X, f, n):
    """
    X : le domaine
    f : la fonction (sur scalaire)
    n : le degré

    Retourne Y, l'approximation de Taylor de f à l'ordre n sur X.
    """
    derivatives=[]
    g=f
    for _ in range(n+1):
        derivatives.append(g(0.0))
        g=grad(g)
    return taylor1(X,derivatives)

def custom(X):
    """retourne 2 * sin(X) + cos(X/4)"""
    return 2*np.sin(X)+np.cos(X/4)


