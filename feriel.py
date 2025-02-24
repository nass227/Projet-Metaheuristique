import random
import numpy as np
from file import SetCoveringProblem # type: ignore

TAUX_MUTATION = 0.1
TAUX_CROISEMENT=0.8
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
        
    # Mutation : Bascule d'un bit (0->1 ou 1->0)
    def mutation(chromosome):
        for i in range(len(chromosome)):
            if random.random() < TAUX_MUTATION:
                chromosome[i] = 1 - chromosome[i]
        return chromosome

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
    
    # REMPLACEMENT
    # CAS 1 : on garde toute la population: rien a faire 
    # CAS 2 : on garde une portion 

    # Fonction de remplacement partiel de la population
    def remplacement_partiel(ancienne_population, nouvelle_population, subsets, universe_size, k,proportion_a_garder=0.5):
        """
        Remplace partiellement la population :
        - Garde un pourcentage `proportion_a_garder` des meilleurs anciens individus.
        - Remplace le reste par des nouveaux individus.
        """
        taille_population = len(ancienne_population)
        taille_a_garder = int(taille_population * proportion_a_garder)  # Nombre d'anciens individus à conserver

        # Trier l'ancienne population par fitness (du meilleur au pire)
        ancienne_population_triee = sorted(ancienne_population, 
                                       key=lambda ind: fitness_function(ind, subsets, universe_size, k), 
                                       reverse=True)
        # Sélection des meilleurs individus
        survivants = ancienne_population_triee[:taille_a_garder]

        # Compléter avec des nouveaux individus choisis aléatoirement
        nouveaux_individus = random.sample(nouvelle_population, taille_population - taille_a_garder)

        # Nouvelle population après remplacement partiel
        return survivants + nouveaux_individus



    def est_solution_valide(solution, k):
        """
        Vérifie si une solution est valide pour le Maximum Covering Problem (MCP).
        
        :param solution: Liste binaire (ex: [1, 0, 1, 0, 1]) indiquant les sous-ensembles choisis.
        :param k: Nombre maximal de sous-ensembles sélectionnés.
        :return: True si valide, False sinon.
        """
        nombre_sous_ensembles = sum(solution)  # Compter le nombre de sous-ensembles sélectionnés
        return nombre_sous_ensembles <= k  # Valide si on prend au plus k sous-ensembles
            
        
    def fitness_function1(solution, subsets, universe_size, k):
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
    
    def evaluation_population(self):
        """ Évalue la fitness de chaque individu dans la population """
        fitness_values = []
        for solution in self.population:
            fitness = self.fitness_function(solution, self.subsets, self.universe_size, self.k)
            fitness_values.append(fitness)
        return fitness_values
    

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






if __name__ == "__main__":
    scp = SetCoveringProblem('test.txt')
    scp.lire_fichier()
    scp.creer_matrice_binaire()
    # print(scp.matrice_binaire[0])
    # print(scp.matrice_binaire[1])
    # child1, child2 = GeneticAlgorithm.uniform_crossover(scp.matrice_binaire[0],scp.matrice_binaire[1])
    # print(child1)
    # print(child2)

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

