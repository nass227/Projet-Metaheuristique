import numpy as np

class DFSSolver:
    def __init__(self, subsets, universe_size, k):
        self.subsets = subsets
        self.universe_size = universe_size
        self.k = k
        self.best_solution = None
        self.best_coverage = 0  # Track the maximum coverage
    
    def evaluate_solution(self, selected):
        """Evaluate the coverage of the selected subsets."""
        covered = set()
        for i, chosen in enumerate(selected):
            if chosen:
                covered.update(self.subsets[i])
        coverage = len(covered)
        print(f"Evaluating solution: {selected} -> Coverage: {coverage}")
        return coverage
    
    def dfs(self, index=0, selected=None, count=0):
        """Depth-First Search with backtracking."""
        if selected is None:
            selected = [0] * len(self.subsets)
        
        # print(f"DFS call -> index: {index}, selected: {selected}, count: {count}")
        
        # Stop if we have considered all subsets (leaf node)
        if index >= len(self.subsets):
            print("reached a feuille")
            if count == self.k:
                coverage = self.evaluate_solution(selected)
                if coverage > self.best_coverage:
                    self.best_coverage = coverage
                    self.best_solution = selected[:]
                    print(f"\n\nNew best solution found! Coverage: {self.best_coverage}\n\n")
            return
        
        # Explore without taking the current subset
        self.dfs(index + 1, selected, count)
        
        # Explore taking the current subset
        selected[index] = 1
        print(f"Taking subset {index}")
        self.dfs(index + 1, selected, count + 1)
        selected[index] = 0  # Backtrack
        print(f"Backtracking from subset {index}")
    
    def solve(self):
        print("Starting DFS Solver...")
        self.dfs()
        print(f"Best solution found: {self.best_solution} with coverage: {self.best_coverage}")
        return self.best_solution, self.best_coverage

# Example usage
if __name__ == "__main__":
    from file import SetCoveringProblem  # Import from your file structure
    
    file_name = "./scp47.txt"
    scp = SetCoveringProblem(file_name)
    scp.lire_fichier()
    scp.creer_matrice_binaire()
    
    subsets = [set(np.where(row == 1)[0]) for row in scp.matrice_binaire]
    universe_size = scp.m
    k = int((2/3) * len(subsets))  # Use the same k as in final.py
    
    solver = DFSSolver(subsets, universe_size, k)
    best_solution, best_coverage = solver.solve()
    
    print("Best solution:", best_solution)
    print("Best coverage:", best_coverage)
