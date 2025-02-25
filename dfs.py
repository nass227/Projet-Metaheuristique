import numpy as np
import time
from file import SetCoveringProblem  # Import from your file structure

class DFSSolver:
    def __init__(self, subsets, universe_size, k):
        self.subsets = subsets
        self.universe_size = universe_size
        self.k = k
        self.best_solution = None
        self.best_coverage = 0  # Track the maximum coverage
        self.total_explored = 0  # Count total solutions explored
        self.valid_solutions = 0  # Count valid solutions found
        self.start_time = None  # Track execution time
    
    def evaluate_solution(self, selected):
        """Evaluate the coverage of the selected subsets."""
        covered = set()
        for i, chosen in enumerate(selected):
            if chosen:
                covered.update(self.subsets[i])
        coverage = len(covered)
        return coverage
    
    def dfs(self, index=0, selected=None, count=0, start_time=None, time_limit=10800):
        """Depth-First Search with backtracking, evaluating solutions as soon as they reach k subsets."""
        if selected is None:
            selected = [0] * len(self.subsets)  # Start with no subsets selected
            start_time = time.time()
        
        # Stop if time limit is exceeded
        if time.time() - start_time > time_limit:
            print("Time limit reached. Stopping search.")
            return
        
        self.total_explored += 1
        
        # Evaluate as soon as we reach k selected subsets
        if count == self.k:
            self.valid_solutions += 1
            coverage = self.evaluate_solution(selected)
            print(f"Valid solution found (Coverage: {coverage})")
            if coverage > self.best_coverage:
                self.best_coverage = coverage
                self.best_solution = selected[:]
            return
        
        # Stop if we have considered all subsets
        if index >= len(self.subsets):
            return
        
        # Explore by adding the current subset first
        selected[index] = 1
        self.dfs(index + 1, selected, count + 1, start_time, time_limit)
        
        # Explore without adding the current subset
        selected[index] = 0
        self.dfs(index + 1, selected, count, start_time, time_limit)

    def dfs_version2(self, index=None, selected=None, count=None, start_time=None, time_limit=10800):
        """Depth-First Search starting with all subsets selected and removing subsets from the right."""
        if selected is None:
            selected = [1] * len(self.subsets)  # Start with all subsets selected
            count = len(self.subsets)  # Start with full selection
            start_time = time.time()
            index = len(self.subsets) - 1  # Start from the last index
        
        # Stop if time limit is exceeded
        if time.time() - start_time > time_limit:
            # print("Time limit reached. Stopping search.")
            return
        
        self.total_explored += 1
        
        # Evaluate as soon as we reach k selected subsets
        if count == self.k:
            self.valid_solutions += 1
            coverage = self.evaluate_solution(selected)
            print(f"Valid solution found (Coverage: {coverage})")
            if coverage > self.best_coverage:
                self.best_coverage = coverage
                self.best_solution = selected[:]
            return
        
        # Stop if we have removed all subsets
        if index < 0:  # Now we terminate when reaching the leftmost side
            return
        
        # Explore by removing the current subset
        selected[index] = 0
        self.dfs_version2(index - 1, selected, count - 1, start_time, time_limit)
        
        # Explore without removing the current subset
        selected[index] = 1  # Restore selection before backtracking
        self.dfs_version2(index - 1, selected, count, start_time, time_limit)

    def solve(self, time_limit=10800):
        print("Starting DFS Solver...")
        start_time = time.time()
        self.dfs_version2(start_time=start_time, time_limit=time_limit)
        execution_time = time.time() - start_time
        print(f"DFS Completed in {execution_time / 3600:.2f} hours")
        print(f"Total solutions explored: {self.total_explored}")
        print(f"Valid solutions found: {self.valid_solutions}")
        print(f"Best solution found: {self.best_solution} with coverage: {self.best_coverage}")
        return self.best_solution, self.best_coverage

# Example usage
if __name__ == "__main__":
    # from file import SetCoveringProblem  # Import from your file structure
    
    file_name = "/content/scp47.txt"
    scp = SetCoveringProblem(file_name)
    scp.lire_fichier()
    scp.creer_matrice_binaire()
    
    subsets = [set(np.where(row == 1)[0]) for row in scp.matrice_binaire]
    universe_size = scp.m
    k = int((2/3) * len(subsets))  # Use the same k as in final.py
    
    solver = DFSSolver(subsets, universe_size, k)
    best_solution, best_coverage = solver.solve(time_limit=10800)  # 3-hour limit
    
    print("Best solution:", best_solution)
    print("Best coverage:", best_coverage)
