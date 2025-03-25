# ShelfOptimizationUsingGeneticAlgorithm
This script uses a Genetic Algorithm (GA) to optimize product placement on shelves while satisfying various constraints.
# Process
# Shelves & Products: 
Represented using Python dictionaries.
# Generate Initial Population: 
Randomly assign products to valid shelves while minimizing capacity violations.
# Evaluate Fitness: 
Solutions are scored based on penalties for weight limits, category mismatches, refrigeration needs, hazardous materials, and accessibility issues.
# Optimize with GA: 
Uses tournament selection, crossover, and mutation to refine solutions across generations.
# Constraint Repair:
A repair function ensures valid shelving after mutations.
