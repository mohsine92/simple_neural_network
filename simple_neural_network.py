# Importation des bibliothèques nécessaires
import numpy as np # Pour les calculs mathématiques et matrices
import matplotlib.pyplot as plt  # Pour créer les graphiques
from typing import List, Tuple   # Pour indiquer les types des variables
import random     



def plot_results(X, y, nn, title: str = "Résultats"):
    """
    VISUALISATION DES RÉSULTATS
    
    Crée de beaux graphiques pour voir comment le réseau a appris
    
    Args:
        X: Données d'entrée
        y: Vraies classes
        nn: Réseau de neurones entraîné
        title: Titre des graphiques
    """
    # Création d'une figure avec 2 graphiques côte à côte
    plt.figure(figsize=(12, 4))
    
    # ========== GRAPHIQUE 1: DATASET ET PRÉDICTIONS ==========
    plt.subplot(1, 2, 1)  # 1 ligne, 2 colonnes, graphique 1
    
    # Faire des prédictions avec notre réseau
    predictions = nn.predict(X)
    
    # COLORATION DES POINTS SELON LEUR VRAIE CLASSE
    # Rouge pour classe 0, bleu pour classe 1
    colors = ['red' if label[0] > 0.5 else 'blue' for label in y]
    plt.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.7, s=100)
    
    # AFFICHAGE DES PRÉDICTIONS DU RÉSEAU
    # À côté de chaque point, on écrit ce que le réseau a prédit
    for i, (point, pred) in enumerate(zip(X, predictions)):
        plt.annotate(f'{pred[0]:.2f}',     # Texte à afficher (prédiction)
                    (point[0], point[1]),  # Position du point
                    xytext=(5, 5),         # Décalage du texte
                    textcoords='offset points', 
                    fontsize=8)
    
    # Décoration du graphique
    plt.title(f'{title} - Dataset et Prédictions')
    plt.xlabel('x1')  # Axe horizontal
    plt.ylabel('x2')  # Axe vertical
    plt.grid(True, alpha=0.3)  # Grille en transparence
    
    # ========== GRAPHIQUE 2: FRONTIÈRE DE DÉCISION ==========
    # Seulement pour les données 2D (x1, x2)
    if X.shape[1] == 2:
        plt.subplot(1, 2, 2)  # 1 ligne, 2 colonnes, graphique 2
        
        # CRÉATION D'UNE GRILLE POUR VOIR COMMENT LE RÉSEAU "VOIT" L'ESPACE
        h = 0.1  # Résolution de la grille
        
        # Limites de l'espace à explorer
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        
        # Création de la grille de points
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # PRÉDICTION SUR CHAQUE POINT DE LA GRILLE
        # On va demander au réseau ce qu'il pense de chaque petit carré
        mesh_points = np.c_[xx.ravel(), yy.ravel()]  # Conversion en liste de points
        Z = nn.predict(mesh_points)                   # Prédiction du réseau
        Z = Z.reshape(xx.shape)                       # Remise en forme de grille
        
        # AFFICHAGE DE LA FRONTIÈRE DE DÉCISION
        # Couleur = probabilité que le point soit de classe 1
        # Bleu = proche de 0 (classe 0), Rouge = proche de 1 (classe 1)
        plt.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='RdBu')
        plt.colorbar(label='Probabilité')  # Légende des couleurs
        
        # AFFICHAGE DES POINTS DU DATASET ORIGINAL
        plt.scatter(X[:, 0], X[:, 1], c=colors, edgecolors='black', s=100)
        
        # Décoration du graphique
        plt.title(f'{title} - Frontière de décision')
              

class SimpleNeuralNetwork:
    """
    RÉSEAU DE NEURONES SIMPLE FAIT MAISON
    
    Imagine un cerveau artificiel avec plusieurs couches :
    - Couche d'entrée : reçoit les données (comme tes yeux qui voient)
    - Couches cachées : traitent l'information (comme ton cerveau qui réfléchit)
    - Couche de sortie : donne la réponse (comme ta bouche qui parle)
    """
    
    def __init__(self, layers: List[int], learning_rate: float = 0.1):
        """
        CONSTRUCTEUR : Crée un nouveau réseau de neurones
        
        Args:
            layers: Liste qui décrit la forme du réseau
                   Ex: [2, 4, 1] = 2 entrées → 4 neurones cachés → 1 sortie
            learning_rate: Vitesse d'apprentissage (entre 0 et 1)
                          Trop haut = apprend trop vite et fait n'importe quoi
                          Trop bas = apprend trop lentement
        """
        # Sauvegarde des paramètres du réseau
        self.layers = layers                    # [2, 4, 1] par exemple
        self.learning_rate = learning_rate      # 0.1 par exemple
        self.num_layers = len(layers)           # 3 dans l'exemple ci-dessus
        
        # CRÉATION DES POIDS ET BIAIS
        # Les poids = importance de chaque connexion entre neurones
        # Les biais = tendance naturelle de chaque neurone
        self.weights = []  # Liste qui contiendra tous les poids
        self.biases = []   # Liste qui contiendra tous les biais
        
        # Boucle pour créer les connexions entre chaque couche
        for i in range(self.num_layers - 1):  # -1 car on connecte couche i à couche i+1
            
            # INITIALISATION INTELLIGENTE DES POIDS (méthode Xavier)
            # Au lieu de mettre des nombres complètement aléatoires,
            # on les ajuste selon la taille de la couche pour éviter
            # que les neurones soient trop excités ou trop endormis
            w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i])
            
            # Les biais commencent à 0 (neutre)
            b = np.zeros((1, layers[i+1]))
            
            # Ajout à nos listes
            self.weights.append(w)
            self.biases.append(b)
    
    def sigmoid(self, x):
        """
        FONCTION D'ACTIVATION SIGMOÏDE
        
        Transforme n'importe quel nombre en valeur entre 0 et 1
        C'est comme un interrupteur graduel :
        - Nombre très négatif (-10) → proche de 0 (éteint)
        - Nombre proche de 0 → proche de 0.5 (indécis)
        - Nombre très positif (10) → proche de 1 (allumé)
        
        Formule mathématique : 1 / (1 + e^(-x))
        """
        # Protection contre les nombres trop grands qui feraient planter l'ordinateur
        x = np.clip(x, -500, 500)  # Limite x entre -500 et 500
        
        # Calcul de la sigmoïde
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """
        DÉRIVÉE DE LA SIGMOÏDE
        
        Nécessaire pour la rétropropagation (apprentissage)
        Indique "à quelle vitesse" la sigmoïde change
        
        Si x = 0.5 (indécis), la dérivée est maximale (0.25)
        Si x = 0 ou 1 (sûr), la dérivée est proche de 0
        """
        return x * (1 - x)  # Formule simplifiée de la dérivée
    
    def forward_propagation(self, X):
        """
        PROPAGATION AVANT - Le réseau fait une prédiction
        
        C'est comme suivre le chemin d'une information dans votre cerveau :
        1. Vos yeux voient quelque chose (entrée)
        2. L'info passe par plusieurs zones du cerveau (couches cachées)
        3. Vous prenez une décision (sortie)
        
        Args:
            X: Données d'entrée (ex: [[0,1], [1,0]] pour XOR)
        
        Returns: 
            Liste des activations de chaque couche
        """
        # On commence avec les données d'entrée
        activations = [X]  # activations[0] = données d'entrée
        
        # Pour chaque couche du réseau (sauf la première qui est l'entrée)
        for i in range(self.num_layers - 1):
            
            # ÉTAPE 1: CALCUL DE LA SOMME PONDÉRÉE
            # Chaque neurone fait une somme des entrées multiplié par les poids
            # Plus le poids est grand, plus l'entrée est importante
            z = np.dot(activations[i], self.weights[i]) + self.biases[i]
            # Exemple: si entrée=[0,1], poids=[[0.5, 0.3], [0.2, 0.8]], biais=[0.1, 0.1]
            # Alors z = [0*0.5 + 1*0.2, 0*0.3 + 1*0.8] + [0.1, 0.1] = [0.3, 0.9]
            
            # ÉTAPE 2: APPLICATION DE LA FONCTION D'ACTIVATION
            # Transformation de z en valeur entre 0 et 1
            a = self.sigmoid(z)
            # Exemple: sigmoid([0.3, 0.9]) = [0.57, 0.71] environ
            
            # ÉTAPE 3: SAUVEGARDE POUR LA COUCHE SUIVANTE
            activations.append(a)
        
        return activations  # Contient l'activation de chaque couche
    
    def backward_propagation(self, X, y, activations):
        """
        RÉTROPROPAGATION - Le réseau apprend de ses erreurs
        
        Imagine que vous lancez une fléchette et ratez la cible :
        1. Vous calculez à quel point vous avez raté (erreur)
        2. Vous analysez quel muscle a mal bougé (propagation de l'erreur)
        3. Vous ajustez votre technique pour le prochain lancer (mise à jour des poids)
        
        Args:
            X: Données d'entrée
            y: Bonnes réponses attendues
            activations: Résultats de chaque couche (de forward_propagation)
        """
        m = X.shape[0]  # Nombre d'exemples dans notre lot de données
        
        # ÉTAPE 1: CALCUL DE L'ERREUR DE SORTIE
        # "À quel point ma prédiction était fausse ?"
        output_error = activations[-1] - y  # activations[-1] = dernière couche (sortie)
        # Exemple: si j'ai prédit 0.8 et la vraie réponse est 1.0, erreur = -0.2
        
        # ÉTAPE 2: CALCUL DU GRADIENT DE SORTIE
        # "Dans quelle direction et à quelle vitesse corriger ?"
        # On multiplie l'erreur par la dérivée de sigmoid pour avoir la "pente"
        deltas = [output_error * self.sigmoid_derivative(activations[-1])]
        
        # ÉTAPE 3: RÉTROPROPAGATION DE L'ERREUR
        # On remonte l'erreur couche par couche, de la sortie vers l'entrée
        # Chaque neurone reçoit sa part de responsabilité dans l'erreur
        for i in range(self.num_layers - 2, 0, -1):  # De l'avant-dernière couche à la première cachée
            
            # L'erreur de cette couche = erreur de la couche suivante × poids de connexion
            # C'est comme dire "si le neurone suivant s'est trompé, 
            # dans quelle mesure c'est ma faute ?"
            error = deltas[-1].dot(self.weights[i].T)  # .T = transposée de la matrice
            
            # Calcul du gradient pour cette couche
            delta = error * self.sigmoid_derivative(activations[i])
            deltas.append(delta)
        
        # On inverse l'ordre car on a calculé de la fin vers le début
        deltas.reverse()
        
        # ÉTAPE 4: MISE À JOUR DES POIDS ET BIAIS
        # "Maintenant qu'on sait qui est responsable, on corrige !"
        for i in range(self.num_layers - 1):
            
            # Mise à jour des poids
            # Règle: poids -= learning_rate × gradient
            # Si gradient > 0, on diminue le poids
            # Si gradient < 0, on augmente le poids
            self.weights[i] -= self.learning_rate * activations[i].T.dot(deltas[i]) / m
            
            # Mise à jour des biais (plus simple)
            self.biases[i] -= self.learning_rate * np.sum(deltas[i], axis=0, keepdims=True) / m
    
    def train(self, X, y, epochs: int = 1000, verbose: bool = True):
        """
        ENTRAÎNEMENT DU RÉSEAU
        
        C'est comme apprendre à jouer au piano :
        1. Vous essayez de jouer une mélodie (forward propagation)
        2. Vous écoutez les fausses notes (calcul de l'erreur)
        3. Vous ajustez la position de vos doigts (backward propagation)
        4. Vous répétez jusqu'à jouer parfaitement (epochs)
        
        Args:
            X: Données d'entrée pour l'entraînement
            y: Réponses correctes attendues
            epochs: Nombre de fois qu'on répète l'entraînement
            verbose: Si True, affiche le progrès
        
        Returns:
            Liste des erreurs à chaque epoch (pour faire des graphiques)
        """
        losses = []  # Pour sauvegarder l'évolution de l'erreur
        
        # Boucle d'entraînement principale
        for epoch in range(epochs):
            
            # ÉTAPE 1: PRÉDICTION (Forward propagation)
            # Le réseau regarde les données et fait sa meilleure prédiction
            activations = self.forward_propagation(X)
            
            # ÉTAPE 2: CALCUL DE L'ERREUR (Loss)
            # On mesure à quel point le réseau s'est trompé
            # MSE = Mean Squared Error (erreur quadratique moyenne)
            # Plus l'erreur est proche de 0, mieux c'est !
            loss = np.mean((activations[-1] - y) ** 2)
            losses.append(loss)  # Sauvegarde pour les graphiques
            
            # ÉTAPE 3: APPRENTISSAGE (Backward propagation)
            # Le réseau apprend de ses erreurs et ajuste ses poids
            self.backward_propagation(X, y, activations)
            
            # ÉTAPE 4: AFFICHAGE DU PROGRÈS
            # Tous les 100 epochs, on affiche où on en est
            if verbose and epoch % 100 == 0:
                print(f"Époque {epoch:4d} - Perte: {loss:.6f}")
        
        return losses  # Retourne l'historique des erreurs
    
    def predict(self, X):
        """
        PRÉDICTION SUR DE NOUVELLES DONNÉES
        
        Une fois que le réseau a appris, on peut lui donner de nouvelles données
        qu'il n'a jamais vues et il fera sa prédiction
        
        Args:
            X: Nouvelles données à prédire
            
        Returns:
            Prédictions du réseau (valeurs entre 0 et 1)
        """
        # On fait juste une forward propagation, sans apprentissage
        activations = self.forward_propagation(X)
        return activations[-1]  # Retourne seulement la sortie finale
    
    def accuracy(self, X, y, threshold: float = 0.5):
        """
        CALCUL DE LA PRÉCISION
        
        Mesure le pourcentage de bonnes prédictions
        
        Args:
            X: Données d'entrée
            y: Vraies réponses
            threshold: Seuil de décision (0.5 par défaut)
                      Si prédiction > 0.5 → classe 1
                      Si prédiction ≤ 0.5 → classe 0
        
        Returns:
            Précision entre 0 et 1 (1 = 100% de bonnes réponses)
        """
        # Faire les prédictions
        predictions = self.predict(X)
        
        # Convertir les prédictions en classes (0 ou 1)
        predicted_classes = (predictions > threshold).astype(int)
        
        # Convertir les vraies réponses en classes
        actual_classes = (y > threshold).astype(int)
        
        # Calculer le pourcentage de bonnes prédictions
        return np.mean(predicted_classes == actual_classes)


def create_xor_dataset():
    """
    CRÉATION DU DATASET XOR
    
    XOR (ou exclusif) est un problème classique en intelligence artificielle
    C'est comme un interrupteur à 2 boutons :
    - Si aucun bouton n'est pressé → lumière éteinte (0)
    - Si un seul bouton est pressé → lumière allumée (1)  
    - Si les deux boutons sont pressés → lumière éteinte (0)
    
    Le défi : impossible de séparer les cas avec une ligne droite !
    Il faut un réseau avec des couches cachées.
    
    Returns:
        X: Données d'entrée [[0,0], [0,1], [1,0], [1,1]]
        y: Réponses attendues [[0], [1], [1], [0]]
    """
    # Les 4 cas possibles avec 2 entrées binaires
    X = np.array([[0, 0],    # Cas 1: 0 XOR 0 = 0
                  [0, 1],    # Cas 2: 0 XOR 1 = 1
                  [1, 0],    # Cas 3: 1 XOR 0 = 1
                  [1, 1]])   # Cas 4: 1 XOR 1 = 0
    
    # Les réponses correspondantes
    y = np.array([[0],       # Réponse cas 1
                  [1],       # Réponse cas 2
                  [1],       # Réponse cas 3
                  [0]])      # Réponse cas 4
    
    return X, y


def create_spiral_dataset(n_points: int = 100):
    """
    CRÉATION D'UN DATASET EN SPIRALE
    
    Problème plus complexe que XOR : deux spirales entrelacées
    Une spirale rouge (classe 0) et une spirale bleue (classe 1)
    
    C'est comme essayer de séparer deux spaghettis entortillés !
    Test parfait pour voir si notre réseau peut apprendre des formes complexes
    
    Args:
        n_points: Nombre total de points à générer
        
    Returns:
        X: Coordonnées des points [(x1,y1), (x2,y2), ...]
        y: Classes des points [0, 1, 0, 1, ...]
    """
    X = []  # Liste des coordonnées
    y = []  # Liste des classes
    
    # Génération de 2 spirales (une par classe)
    for class_num in range(2):  # class_num = 0 puis 1
        
        # CRÉATION DES PARAMÈTRES DE LA SPIRALE
        # r = rayon qui grandit progressivement (spirale vers l'extérieur)
        r = np.linspace(0.1, 1, n_points // 2)  # De 0.1 à 1.0
        
        # t = angle qui tourne (spirale qui tourne)
        # Chaque spirale est décalée pour qu'elles s'entremêlent
        t = np.linspace(class_num * np.pi, (class_num + 2) * np.pi, n_points // 2)
        
        # GÉNÉRATION DES POINTS
        for i in range(n_points // 2):
            # Coordonnées polaires → coordonnées cartésiennes
            x1 = r[i] * np.cos(t[i]) + random.gauss(0, 0.1)  # + bruit aléatoire
            x2 = r[i] * np.sin(t[i]) + random.gauss(0, 0.1)  # + bruit aléatoire
            
            # Ajout du point à notre dataset
            X.append([x1, x2])
            y.append([class_num])  # 0 pour première spirale, 1 pour seconde
    
    return np.array(X), np.array(y)


def plot_results(X, y, nn, title: str = "Résultats"):
    """Visualise les résultats du réseau"""
    plt.figure(figsize=(12, 4))
    
    # Graphique 1: Dataset et prédictions
    plt.subplot(1, 2, 1)
    predictions = nn.predict(X)
    
    # Points colorés selon les vraies classes
    colors = ['red' if label[0] > 0.5 else 'blue' for label in y]
    plt.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.7, s=100)
    
    # Affiche les prédictions comme texte
    for i, (point, pred) in enumerate(zip(X, predictions)):
        plt.annotate(f'{pred[0]:.2f}', (point[0], point[1]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.title(f'{title} - Dataset et Prédictions')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True, alpha=0.3)
    
    # Graphique 2: Frontière de décision (pour datasets 2D)
    if X.shape[1] == 2:
        plt.subplot(1, 2, 2)
        
        # Crée une grille pour visualiser la frontière
        h = 0.1
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Prédit sur chaque point de la grille
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = nn.predict(mesh_points)
        Z = Z.reshape(xx.shape)
        
        # Affiche la frontière de décision
        plt.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='RdBu')
        plt.colorbar(label='Probabilité')
        
        # Affiche les points du dataset
        plt.scatter(X[:, 0], X[:, 1], c=colors, edgecolors='black', s=100)
        
        plt.title(f'{title} - Frontière de décision')
        plt.xlabel('x1')
        plt.ylabel('x2')
    
    plt.tight_layout()
    plt.show()


def main():
    """Fonction principale pour tester le réseau"""
    print("=== Test du Réseau de Neurones Simple ===\n")
    
    # Test 1: Problème XOR
    print("1. Test sur le problème XOR")
    print("-" * 30)
    
    X_xor, y_xor = create_xor_dataset()
    print("Dataset XOR:")
    for i in range(len(X_xor)):
        print(f"  {X_xor[i]} -> {y_xor[i][0]}")
    
    # Crée et entraîne le réseau pour XOR
    nn_xor = SimpleNeuralNetwork([2, 4, 1], learning_rate=1.0)
    losses_xor = nn_xor.train(X_xor, y_xor, epochs=2000, verbose=True)
    
    print(f"\nPrécision finale: {nn_xor.accuracy(X_xor, y_xor):.2%}")
    print("\nPrédictions finales:")
    predictions = nn_xor.predict(X_xor)
    for i in range(len(X_xor)):
        print(f"  {X_xor[i]} -> {predictions[i][0]:.4f} (attendu: {y_xor[i][0]})")
    
    # Test 2: Dataset spirale (plus complexe)
    print("\n\n2. Test sur un dataset en spirale")
    print("-" * 35)
    
    X_spiral, y_spiral = create_spiral_dataset(200)
    nn_spiral = SimpleNeuralNetwork([2, 10, 8, 1], learning_rate=0.5)
    losses_spiral = nn_spiral.train(X_spiral, y_spiral, epochs=1000, verbose=True)
    
    print(f"\nPrécision sur spirale: {nn_spiral.accuracy(X_spiral, y_spiral):.2%}")
    
    # Visualisation des résultats
    plot_results(X_xor, y_xor, nn_xor, "XOR")
    plot_results(X_spiral, y_spiral, nn_spiral, "Spirale")
    
    # Graphique des courbes de perte
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