import numpy as np
import random
from file import SetCoveringProblem # type: ignore
import time



class GeneticAlgorithm:
    def __init__(self, subsets, universe_size, k, taux_mutation=0.05, taille_population=50, n_generations=50 ,type_croisement='deux_points',   # 'deux_points' ou 'uniforme'
                 type_selection='tournoi' ,dev_population='remplacement'):
        self.subsets = subsets
        self.universe_size = universe_size
        self.k = k
        self.taux_mutation = taux_mutation
        self.taille_population = taille_population
        self.n_generations = n_generations
        self.type_croisement = type_croisement
        self.type_selection = type_selection
        self.dev_population = dev_population
        self.population = []



    def initialisation_population(self):
        """ 
        Génère une population initiale de solutions aléatoires 
        avec exactement k éléments à 1 pour respecter la contrainte. 
        """
        self.population = []
        for _ in range(self.taille_population):
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
            print(f"Individu {i} couvre {sous_ensembles_couverts} sous-ensembles.") #pour s'assurer qu'il ne depasse pas K


    
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



    def selection_tournoi(self, fitness_values):
        """ Sélectionne les parents en utilisant la méthode de sélection par tournoi """
        parents = []
        for _ in range(len(self.population)): #retoune autant de parents que d'individus
            candidats = random.sample(list(zip(self.population, fitness_values)), k=3)
            parents.append(max(candidats, key=lambda x: x[1])[0])  # Sélectionne le meilleur du tournoi
        return parents
    



    # Croisement à un point
    @staticmethod
    def two_point_crossover(parent1, parent2):
        # Choisir deux points de croisement aléatoires
        point1 = random.randint(1, len(parent1) - 2)
        point2 = random.randint(point1 + 1, len(parent1) - 1)
        
        # Création des enfants
        child1 = np.concatenate((parent1[:point1], parent2[point1:point2], parent1[point2:]))
        child2 = np.concatenate((parent2[:point1], parent1[point1:point2], parent2[point2:]))
        
        return child1, child2
    


       

    def mutation(self, child, mutation_rate):
        """
        Applique une mutation avec un taux donné tout en respectant la contrainte d'avoir exactement k sous-ensembles sélectionnés.
        
        Parameters:
            - child: Liste représentant un individu
            - mutation_rate: Probabilité de mutation d'un gène (par défaut 1%)
        """
        # Parcourir chaque gène (bit) de l'individu
        for index in range(len(child)):
            # Appliquer la mutation selon le taux de mutation
            if random.random() < mutation_rate:
                # Inverser le bit
                child[index] = 1 - child[index]
        
        # Vérifier le nombre de 1 après la mutation
        nb_ones = sum(child)
        
        # Ajuster pour respecter la contrainte de k sous-ensembles
        if nb_ones > self.k:
            # Trop de 1 : en enlever aléatoirement jusqu'à avoir exactement k
            indices_a_enlever = [i for i in range(len(child)) if child[i] == 1]
            while sum(child) > self.k:
                index_a_enlever = random.choice(indices_a_enlever)
                child[index_a_enlever] = 0
                indices_a_enlever.remove(index_a_enlever)
        
        elif nb_ones < self.k:
            # Pas assez de 1 : en ajouter aléatoirement jusqu'à avoir exactement k
            indices_a_ajouter = [i for i in range(len(child)) if child[i] == 0]
            while sum(child) < self.k:
                index_a_ajouter = random.choice(indices_a_ajouter)
                child[index_a_ajouter] = 1
                indices_a_ajouter.remove(index_a_ajouter)
        
        return child


   # Fonction de remplacement partiel de la population
    def remplacement_partiel(self, ancienne_population, nouvelle_population, proportion_a_garder=0.5):
        """
        Remplace partiellement la population :
        - Garde un pourcentage `proportion_a_garder` des meilleurs anciens individus.
        - Remplace le reste par des nouveaux individus.
        
        Si la nouvelle population est trop petite, elle est complétée aléatoirement avec des individus existants.
        """
        taille_population = len(ancienne_population)
        taille_a_garder = int(taille_population * proportion_a_garder)  # Nombre d'anciens individus à conserver
        taille_a_remplacer = taille_population - taille_a_garder  # Nombre d'individus à remplacer

        # Trier l'ancienne population par fitness (du meilleur au pire)
        ancienne_population_triee = sorted(ancienne_population, key=lambda sol: self.evaluer_fitness(sol), reverse=True)

        # Sélection des meilleurs individus
        survivants = ancienne_population_triee[:taille_a_garder]

        # Vérifier si la nouvelle population est suffisante
        if len(nouvelle_population) < taille_a_remplacer:
            # Compléter avec des individus existants pour éviter une erreur
            individus_complementaires = random.choices(ancienne_population, k=taille_a_remplacer - len(nouvelle_population))
            nouvelle_population.extend(individus_complementaires)

        # Sélection aléatoire des nouveaux individus pour compléter
        nouveaux_individus = random.sample(nouvelle_population, taille_a_remplacer)

        # Nouvelle population après remplacement partiel
        return survivants + nouveaux_individus

    
    
    def evaluer_fitness(self, solution):
        return self.fitness_function(solution, self.subsets, self.universe_size, self.k)
    


    def evolution(self):
        """ Gère le cycle complet de sélection, croisement, mutation et remplacement """
        self.initialisation_population()
        
        fitness_historique = []  # Historique des fitness

        for generation in range(self.n_generations):
            fitness_values = self.evaluation_population()
            parents = self.selection_tournoi(fitness_values)
        
            new_population = []

            # Croisement et mutation
            for i in range(0, len(parents) - 1, 2):
                parent1 = parents[i]
                parent2 = parents[i + 1]

                #croisement
                child1, child2 = self.two_point_crossover(parent1, parent2)
                child1 = self.mutation(child1,self.taux_mutation)
                child2 = self.mutation(child2,self.taux_mutation)

                new_population.extend([child1, child2])

            # Si le nombre de parents est impair, le dernier parent est ajouté sans croisement
            if len(parents) % 2 == 1:
                new_population.append(parents[-1])

            
            #  Appel de remplacement_partiel pour combiner ancienne et nouvelle population
            self.population = self.remplacement_partiel(self.population, new_population, proportion_a_garder=0.5)
            
         

            # Meilleure solution de la génération actuelle
            meilleur_individu = max(self.population, key=lambda sol: self.fitness_function(sol, self.subsets, self.universe_size, self.k))
            meilleure_fitness = self.fitness_function(meilleur_individu, self.subsets, self.universe_size, self.k)
            
            # Ajouter à l'historique des fitness
            fitness_historique.append(meilleure_fitness)

            # Afficher la meilleure solution de chaque génération
            print(f"Génération {generation + 1}: Meilleure Fitness = {meilleure_fitness}")
        
        # Retourner la meilleure solution trouvée et l'historique des fitness
        return meilleur_individu, fitness_historique
    
    def printsol(self, solution, ens):
        sol = []
        obj = set()  

        for i in range(len(solution)):
            if solution[i] == 1:  
                sol.append(i + 1)  
                obj.update(ens[i])  

        return sol, obj  







if __name__ == "__main__":
    # Lecture du problème de couverture d'ensemble
    scp = SetCoveringProblem('test.txt')
    scp.lire_fichier()
    scp.creer_matrice_binaire()

    # Création des sous-ensembles pour l'algorithme génétique
    subsets =scp.contraintes
    universe_size = scp.m
    k = int((2/3) * len(subsets))  # k est égal à 2/3 du nombre de sous-ensembles

    ga = GeneticAlgorithm(subsets=subsets, universe_size=universe_size, k=k)

    
    best_solution, fitness_history = ga.evolution()
    best_fitness = max(fitness_history)  # Meilleure fitness obtenue

    print("la valeur de K :",k)
    print(f"Fitness: {best_fitness}\n")
    print(f"Solution: {best_solution}")  # Afficher la meilleure solution trouvée

    sol, obj = ga.printsol(best_solution, subsets)

    print("La liste des ensembles séléctionnée :", sol)
    print("Taille de l'ensemble :", len(sol))
    print("Objets couverts :", obj)
    print("Nombre total d'objets couverts :", len(obj))



    




