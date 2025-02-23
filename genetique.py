import random
import numpy as np


TAUX_CROISEMENT=0.8

class GeneticAlgorithm:
    def __init__(self):
        pass

    # Croisement à un point
    def one_point_crossover(parent1, parent2):
        # choisir de faire un croisement ou pas aléatoirement
        if random.random() < TAUX_CROISEMENT:
            point = random.randint(1, len(parent1) - 1) #choisir un point de croisement aléatoirement
            child1 = np.concatenate((parent1[:point], parent2[point:]))
            child2 = np.concatenate((parent2[:point], parent1[point:]))
            return child1, child2
        else:
            return parent1[:], parent2[:]
        
    def two_point_crossover(parent1, parent2):
        # choisir de faire un croisement ou pas aléatoirement
        if random.random() < TAUX_CROISEMENT:
            point1, point2 = sorted(random.sample(range(1, len(parent1) - 1), 2))  #selectionne aléatoirement 2 points de croisement dans l'ordre croissant : 2fois range(1, len(parent1) - 1)
            child1 = np.concatenate((parent1[:point1], parent2[point1:point2], parent1[point2:]))
            child2 = np.concatenate((parent2[:point1], parent1[point1:point2], parent2[point2:]))
            return child1, child2
        else:
            return parent1[:], parent2[:]
        
    def uniform_crossover(parent1, parent2, prob=0.5):
        #Effectue un croisement uniforme avec probabilité de sélection de parent2 
        if random.random() < TAUX_CROISEMENT:
            child1 = []
            child2 = []
            for i in range(len(parent1)):
                if random.random() < prob:  # Probabilité de choisir un gène du parent2
                    child1.append(parent2[i])
                    child2.append(parent1[i])
                else:
                    child1.append(parent1[i])
                    child2.append(parent2[i])
            
            return np.array(child1), np.array(child2)  # Retourne des vecteurs NumPy
        else:
            return parent1[:], parent2[:]
        
    
    def fitness_function(solution, subsets, universe_size, k):
        """
        Arguments :
        - solution (list[int]): Vecteur binaire représentant les sous-ensembles sélectionnés.
        - subsets (list[set]): Liste des sous-ensembles disponibles.
        - universe_size (int): Nombre total d'éléments dans l'univers.
        - k (int): Nombre exact de sous-ensembles à sélectionner.
        Returns:
        - float: Fitness normalisée entre 0 et 1 (ou pénalité si solution invalide).
        """
        # Vérifier si la solution respecte la contrainte k
        if sum(solution) != k:
            return -1  # Pénalité si le nombre de sous-ensembles sélectionnés n'est pas exactement k
        
        # Couvrir les éléments
        covered_elements = set()
        for i, selected in enumerate(solution):
            if selected == 1:
                covered_elements.update(subsets[i]) #ajouter le sous-ensemble à la liste
        
        # Calcul de la fitness (normalisée entre 0 et 1)
        fitness = len(covered_elements) / universe_size
    
        return fitness


class SetCoveringProblem:
    def __init__(self, nom_fichier):
        self.nom_fichier = nom_fichier
        self.m = None  # Nombre de contraintes (lignes)
        self.n = None  # Nombre de colonnes
        self.couts = None  # Coûts des colonnes
        self.contraintes = None  # Liste des contraintes
        self.matrice_binaire = None  # Matrice binaire des contraintes

    def lire_fichier(self):
        """Lit le fichier et initialise les attributs."""
        with open(self.nom_fichier, 'r') as fichier:
            # Lire la première ligne
            self.m, self.n = map(int, fichier.readline().split())
            
            # Lire les coûts des colonnes
            self.couts = []
            while len(self.couts) < self.n:
                ligne = fichier.readline().strip()
                if not ligne:
                    continue  # Ignorer les lignes vides
                self.couts.extend(map(int, ligne.split()))
            
            # Lire les contraintes
            self.contraintes = []
            for _ in range(self.m):
                # Lire le nombre de colonnes qui couvrent la ligne
                while True:
                    ligne = fichier.readline().strip()
                    if not ligne:
                        continue  # Ignorer les lignes vides
                    nb_colonnes = int(ligne)
                    break
                
                # Lire les indices des colonnes
                colonnes = []
                while len(colonnes) < nb_colonnes:
                    ligne = fichier.readline().strip()
                    if not ligne:
                        continue  # Ignorer les lignes vides
                    colonnes.extend(map(int, ligne.split()))
                
                self.contraintes.append(colonnes)

    def creer_matrice_binaire(self):
        """Crée la matrice binaire des contraintes."""
        # Initialiser une matrice de zéros de taille (m, n)
        self.matrice_binaire = np.zeros((self.m, self.n), dtype=int)
        
        # Remplir la matrice
        for i, colonnes in enumerate(self.contraintes):
            for j in colonnes:
                self.matrice_binaire[i, j - 1] = 1  # Les indices commencent à 0 en Python

    def afficher_informations(self):
        """Affiche les informations du problème."""
        print(f"Nombre de lignes (m) : {self.m}")
        print(f"Nombre de colonnes (n) : {self.n}")
        print(f"Coûts des colonnes : {self.couts[:10]}...")  # Affiche les 10 premiers coûts
        print(f"Première contrainte dans la matrice : {self.matrice_binaire[0]}")
        print(f"Première contrainte dans la liste : {self.contraintes[0]}")

if __name__ == "__main__":
    scp = SetCoveringProblem('test.txt')
    scp.lire_fichier()
    scp.creer_matrice_binaire()
    print(scp.matrice_binaire[0])
    print(scp.matrice_binaire[1])
    child1, child2 = GeneticAlgorithm.uniform_crossover(scp.matrice_binaire[0],scp.matrice_binaire[1])
    print(child1)
    print(child2)
