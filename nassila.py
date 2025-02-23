import random
TAUX_MUTATION = 0.1

# Mutation : Bascule d'un bit (0->1 ou 1->0)
def mutation(chromosome):
    for i in range(len(chromosome)):
        if random.random() < TAUX_MUTATION:
            chromosome[i] = 1 - chromosome[i]
    return chromosome

# REMPLACEMENT
# CAS 1 : on garde toute la population: rien a faire 
# CAS 2 : on garde une portion 

# Fonction de remplacement partiel de la population
def remplacement_partiel(ancienne_population, nouvelle_population, proportion_a_garder=0.5):
    """
    Remplace partiellement la population :
    - Garde un pourcentage `proportion_a_garder` des meilleurs anciens individus.
    - Remplace le reste par des nouveaux individus.
    """
    taille_population = len(ancienne_population)
    taille_a_garder = int(taille_population * proportion_a_garder)  # Nombre d'anciens individus à conserver

    # Trier l'ancienne population par fitness (du meilleur au pire)
    ancienne_population_triee = sorted(ancienne_population, key=evaluer_fitness, reverse=True)

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
