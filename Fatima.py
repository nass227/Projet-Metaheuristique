import numpy as np
import random

TAUX_MUTATION = 0.01
TAILLE_POPULATION = 10
N_GENERATIONS = 10

class GeneticAlgorithm:
    def __init__(self, subsets, universe_size, k):
        self.subsets = subsets
        self.universe_size = universe_size
        self.k = k
        self.population = []

    def initialisation_population(self):
        """ 
        Génère une population initiale de solutions aléatoires 
        avec exactement k éléments à 1 pour respecter la contrainte. 
        """
        self.population = []
        for _ in range(TAILLE_POPULATION):
            # Créer un vecteur de zéros
            solution = [0] * len(self.subsets)
            
            # Choisir exactement k positions à mettre à 1
            indices = random.sample(range(len(self.subsets)), self.k)
            
            # Mettre à 1 les indices choisis
            for index in indices:
                solution[index] = 1
            
            # Ajouter la solution à la population
            self.population.append(solution)
        
        # Vérification du nombre de sous-ensembles couverts
        for i, sol in enumerate(self.population):
            sous_ensembles_couverts = sum([1 for j in range(len(sol)) if sol[j] == 1])
            print(f"Individu {i} couvre {sous_ensembles_couverts} sous-ensembles.")


    
    def evaluation_population(self):
        """ Évalue la fitness de chaque individu dans la population """
        fitness_values = []
        for solution in self.population:
            fitness = self.fitness_function(solution, self.subsets, self.universe_size, self.k)
            fitness_values.append(fitness)
        return fitness_values
    @staticmethod
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
        fitness = len(covered_elements)
    
        return fitness

    def selection(self, fitness_values):
        """ Sélectionne les parents en utilisant la méthode de sélection par tournoi """
        parents = []
        for _ in range(TAILLE_POPULATION):
            candidats = random.sample(list(zip(self.population, fitness_values)), k=3)
            parents.append(max(candidats, key=lambda x: x[1])[0])  # Sélectionne le meilleur du tournoi
        return parents
    
    def selection_random(self, fitness_values):
        """ Sélectionne les parents aléatoirement sans tenir compte de la fitness """
        parents = []
        for _ in range(TAILLE_POPULATION):
            parent = random.choice(self.population)
            parents.append(parent)
        return parents
    
    def selection_roulette(self, fitness_values):
        """ Sélectionne les parents en utilisant la méthode de sélection par roulette avec normalisation """
        parents = []

        # Normalisation des fitness pour les rendre positives
        min_fitness = min(fitness_values)
        max_fitness = max(fitness_values)
        if max_fitness > min_fitness:  # Éviter la division par zéro
            fitness_values = [(f - min_fitness) / (max_fitness - min_fitness) for f in fitness_values]
        else:
            fitness_values = [1 for _ in fitness_values]  # Si toutes les fitness sont identiques, donner des chances égales

        total_fitness = sum(fitness_values)
        probabilites = [f / total_fitness for f in fitness_values]
        cumulatives = np.cumsum(probabilites)
        
        for _ in range(TAILLE_POPULATION):
            r = random.random()
            for i, cumulative in enumerate(cumulatives):
                if r <= cumulative:
                    parents.append(self.population[i])
                    break
    
        return parents


    # Croisement à un point
    @staticmethod
    def one_point_crossover(parent1, parent2):
        # choisir de faire un croisement ou pas aléatoirement
        
            point = random.randint(1, len(parent1) - 1) #choisir un point de croisement aléatoirement
            child1 = np.concatenate((parent1[:point], parent2[point:]))
            child2 = np.concatenate((parent2[:point], parent1[point:]))
            return child1, child2
       

    def mutation(self, child):
        """ Applique une mutation à un individu """
        for i in range(len(child)):
            if random.random() < TAUX_MUTATION:
                child[i] = 1 - child[i]  # Inverse le bit (0 devient 1 et vice versa)
        return child

    def evolution(self):
        """ Gère le cycle complet de sélection, croisement, mutation et remplacement """
        self.initialisation_population()
        
        for generation in range(N_GENERATIONS):
            fitness_values = self.evaluation_population()
            parents = self.selection_roulette(fitness_values)
            new_population = []

            # Croisement
            for i in range(0, len(parents) - 1, 2):
                parent1 = parents[i]
                parent2 = parents[i + 1]

                child1, child2 = self.one_point_crossover(parent1, parent2)
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)

                new_population.extend([child1, child2])

            # Si le nombre de parents est impair, le dernier parent est ajouté sans croisement
            if len(parents) % 2 == 1:
                new_population.append(parents[-1])

            # Ajouter les nouveaux individus à la population existante
            self.population.extend(new_population)
            
            # Limiter la taille de la population en gardant les meilleurs individus
            self.population = sorted(
                self.population, 
                key=lambda sol: self.fitness_function(sol, self.subsets, self.universe_size, self.k), 
                reverse=True
            )[:TAILLE_POPULATION]

            # Afficher la meilleure solution de chaque génération
            meilleur_individu = max(self.population, key=lambda sol: self.fitness_function(sol, self.subsets, self.universe_size, self.k))
            meilleure_fitness = self.fitness_function(meilleur_individu, self.subsets, self.universe_size, self.k)
            print(f"Génération {generation + 1}: Meilleure Fitness = {meilleure_fitness}")
        
        # Retourner la meilleure solution trouvée
        return max(self.population, key=lambda sol: self.fitness_function(sol, self.subsets, self.universe_size, self.k))


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
    # Lecture du problème de couverture d'ensemble
    scp = SetCoveringProblem('C:/Users/HP/OneDrive/Documents/M1 SII/Meta/Projet/scp41.txt')
    scp.lire_fichier()
    scp.creer_matrice_binaire()

    # Création des sous-ensembles pour l'algorithme génétique
    subsets = [set(np.where(scp.matrice_binaire[:, j] == 1)[0]) for j in range(scp.n)]
    universe_size = scp.m
    k = 5  # Exemple : nombre exact de sous-ensembles à sélectionner

    # Exécution de l'algorithme génétique
    ga = GeneticAlgorithm(subsets, universe_size, k)
    meilleure_solution = ga.evolution()

    # Affichage de la meilleure solution trouvée
    print("\nMeilleure solution trouvée :", meilleure_solution)
    print("Fitness de la meilleure solution :", ga.fitness_function(meilleure_solution, subsets, universe_size, k))
