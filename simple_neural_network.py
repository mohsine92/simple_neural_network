# Importation des bibliothèques nécessaires
import numpy as np # Calculs mathématiques
import matplotlib.pyplot as plt  # Graphiques
from typing import List, Tuple   # Types de variables
import random     



def plot_results(X, y, nn, title: str = "Résultats"):
    """
    Visualise les résultats du réseau
    """
    # Création d'une figure avec 2 graphiques
    plt.figure(figsize=(12, 4))
    
    # Graphique 1: Points et prédictions
    plt.subplot(1, 2, 1)
    
    # Prédictions du réseau
    predictions = nn.predict(X)
    
    # Coloration des points selon leur vraie classe
    colors = ['red' if label[0] > 0.5 else 'blue' for label in y]
    plt.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.7, s=100)
    
    # Affichage des prédictions à côté de chaque point
    for i, (point, pred) in enumerate(zip(X, predictions)):
        plt.annotate(f'{pred[0]:.2f}',
                    (point[0], point[1]),
                    xytext=(5, 5),
                    textcoords='offset points', 
                    fontsize=8)
    
    # Décoration du graphique
    plt.title(f'{title} - Dataset et Prédictions')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True, alpha=0.3)
    
    # Graphique 2: Frontière de décision (données 2D seulement)
    if X.shape[1] == 2:
        plt.subplot(1, 2, 2)
        
        # Création d'une grille pour voir la frontière
        h = 0.1
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Prédiction sur chaque point de la grille
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = nn.predict(mesh_points)
        Z = Z.reshape(xx.shape)
        
        # Affichage de la frontière de décision
        plt.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='RdBu')
        plt.colorbar(label='Probabilité')
        
        # Points du dataset original
        plt.scatter(X[:, 0], X[:, 1], c=colors, edgecolors='black', s=100)
        
        plt.title(f'{title} - Frontière de décision')
              

class SimpleNeuralNetwork:
    """
    Réseau de neurones simple
    """
    
    def __init__(self, layers: List[int], learning_rate: float = 0.1):
        """
        Initialise le réseau
        layers: [2, 4, 1] = 2 entrées → 4 neurones → 1 sortie
        learning_rate: vitesse d'apprentissage
        """
        self.layers = layers
        self.learning_rate = learning_rate
        self.num_layers = len(layers)
        
        # Création des poids et biais
        self.weights = []
        self.biases = []
        
        # Initialisation des connexions entre couches
        for i in range(self.num_layers - 1):
            # Poids aléatoires ajustés selon la taille de couche
            w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i])
            # Biais initialisés à zéro
            b = np.zeros((1, layers[i+1]))
            
            self.weights.append(w)
            self.biases.append(b)
    
    def sigmoid(self, x):
        """
        Fonction d'activation: transforme tout nombre en valeur 0-1
        """
        # Protection contre les nombres trop grands
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """
        Dérivée de la sigmoïde (pour l'apprentissage)
        """
        return x * (1 - x)
    
    def forward_propagation(self, X):
        """
        Propagation avant: le réseau fait une prédiction
        """
        # Début avec les données d'entrée
        activations = [X]
        
        # Passage dans chaque couche
        for i in range(self.num_layers - 1):
            # Calcul: entrée × poids + biais
            z = np.dot(activations[i], self.weights[i]) + self.biases[i]
            # Application de la fonction d'activation
            a = self.sigmoid(z)
            activations.append(a)
        
        return activations
    
    def backward_propagation(self, X, y, activations):
        """
        Rétropropagation: le réseau apprend de ses erreurs
        """
        m = X.shape[0]  # Nombre d'exemples
        
        # Calcul de l'erreur de sortie
        output_error = activations[-1] - y
        
        # Calcul du gradient de sortie
        deltas = [output_error * self.sigmoid_derivative(activations[-1])]
        
        # Rétropropagation de l'erreur couche par couche
        for i in range(self.num_layers - 2, 0, -1):
            error = deltas[-1].dot(self.weights[i].T)
            delta = error * self.sigmoid_derivative(activations[i])
            deltas.append(delta)
        
        deltas.reverse()
        
        # Mise à jour des poids et biais
        for i in range(self.num_layers - 1):
            self.weights[i] -= self.learning_rate * activations[i].T.dot(deltas[i]) / m
            self.biases[i] -= self.learning_rate * np.sum(deltas[i], axis=0, keepdims=True) / m
    
    def train(self, X, y, epochs: int = 1000, verbose: bool = True):
        """
        Entraîne le réseau
        """
        losses = []
        
        for epoch in range(epochs):
            # Prédiction
            activations = self.forward_propagation(X)
            
            # Calcul de l'erreur
            loss = np.mean((activations[-1] - y) ** 2)
            losses.append(loss)
            
            # Apprentissage
            self.backward_propagation(X, y, activations)
            
            # Affichage du progrès
            if verbose and epoch % 100 == 0:
                print(f"Époque {epoch:4d} - Perte: {loss:.6f}")
        
        return losses
    
    def predict(self, X):
        """
        Fait une prédiction
        """
        activations = self.forward_propagation(X)
        return activations[-1]  # Sortie finale
    
    def accuracy(self, X, y, threshold: float = 0.5):
        """
        Calcule la précision (pourcentage de bonnes réponses)
        """
        predictions = self.predict(X)
        predicted_classes = (predictions > threshold).astype(int)
        actual_classes = (y > threshold).astype(int)
        return np.mean(predicted_classes == actual_classes)


def create_xor_dataset():
    """
    Crée le dataset XOR (ou exclusif)
    """
    X = np.array([[0, 0],    # 0 XOR 0 = 0
                  [0, 1],    # 0 XOR 1 = 1
                  [1, 0],    # 1 XOR 0 = 1
                  [1, 1]])   # 1 XOR 1 = 0
    
    y = np.array([[0],
                  [1],
                  [1],
                  [0]])
    
    return X, y


def create_spiral_dataset(n_points: int = 100):
    """
    Crée un dataset en spirale (problème plus complexe)
    """
    X = []
    y = []
    
    # Génération de 2 spirales entremêlées
    for class_num in range(2):
        # Rayon croissant
        r = np.linspace(0.1, 1, n_points // 2)
        # Angle avec décalage pour chaque spirale
        t = np.linspace(class_num * np.pi, (class_num + 2) * np.pi, n_points // 2)
        
        # Génération des points
        for i in range(n_points // 2):
            # Coordonnées + bruit aléatoire
            x1 = r[i] * np.cos(t[i]) + random.gauss(0, 0.1)
            x2 = r[i] * np.sin(t[i]) + random.gauss(0, 0.1)
            
            X.append([x1, x2])
            y.append([class_num])
    
    return np.array(X), np.array(y)


def plot_results(X, y, nn, title: str = "Résultats"):
    """Visualise les résultats"""
    plt.figure(figsize=(12, 4))
    
    # Graphique 1: Dataset et prédictions
    plt.subplot(1, 2, 1)
    predictions = nn.predict(X)
    
    colors = ['red' if label[0] > 0.5 else 'blue' for label in y]
    plt.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.7, s=100)
    
    for i, (point, pred) in enumerate(zip(X, predictions)):
        plt.annotate(f'{pred[0]:.2f}', (point[0], point[1]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.title(f'{title} - Dataset et Prédictions')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True, alpha=0.3)
    
    # Graphique 2: Frontière de décision
    if X.shape[1] == 2:
        plt.subplot(1, 2, 2)
        
        h = 0.1
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = nn.predict(mesh_points)
        Z = Z.reshape(xx.shape)
        
        plt.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='RdBu')
        plt.colorbar(label='Probabilité')
        plt.scatter(X[:, 0], X[:, 1], c=colors, edgecolors='black', s=100)
        
        plt.title(f'{title} - Frontière de décision')
        plt.xlabel('x1')
        plt.ylabel('x2')
    
    plt.tight_layout()
    plt.show()


def main():
    """Fonction principale de test"""
    print("=== Test du Réseau de Neurones Simple ===\n")
    
    # Test 1: XOR
    print("1. Test sur le problème XOR")
    print("-" * 30)
    
    X_xor, y_xor = create_xor_dataset()
    print("Dataset XOR:")
    for i in range(len(X_xor)):
        print(f"  {X_xor[i]} -> {y_xor[i][0]}")
    
    # Entraînement XOR
    nn_xor = SimpleNeuralNetwork([2, 4, 1], learning_rate=1.0)
    losses_xor = nn_xor.train(X_xor, y_xor, epochs=2000, verbose=True)
    
    print(f"\nPrécision finale: {nn_xor.accuracy(X_xor, y_xor):.2%}")
    print("\nPrédictions finales:")
    predictions = nn_xor.predict(X_xor)
    for i in range(len(X_xor)):
        print(f"  {X_xor[i]} -> {predictions[i][0]:.4f} (attendu: {y_xor[i][0]})")
    
    # Test 2: Spirale
    print("\n\n2. Test sur un dataset en spirale")
    print("-" * 35)
    
    X_spiral, y_spiral = create_spiral_dataset(200)
    nn_spiral = SimpleNeuralNetwork([2, 10, 8, 1], learning_rate=0.5)
    losses_spiral = nn_spiral.train(X_spiral, y_spiral, epochs=1000, verbose=True)
    
    print(f"\nPrécision sur spirale: {nn_spiral.accuracy(X_spiral, y_spiral):.2%}")
    
    # Visualisations
    plot_results(X_xor, y_xor, nn_xor, "XOR")
    plot_results(X_spiral, y_spiral, nn_spiral, "Spirale")
    
    # Courbes de perte
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses_xor)
    plt.title('Courbe de perte - XOR')
    plt.xlabel('Époque')
    plt.ylabel('Perte MSE')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(losses_spiral)
    plt.title('Courbe de perte - Spirale')
    plt.xlabel('Époque')
    plt.ylabel('Perte MSE')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
