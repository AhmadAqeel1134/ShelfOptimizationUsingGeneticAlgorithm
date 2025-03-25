import random
import pandas as pd
from collections import defaultdict

# ======================
# ENHANCED DATA STRUCTURES
# ======================

shelves = {
    "S1": {
        "type": "checkout_display",
        "capacity": 8,
        "allowed_categories": ["snacks", "small_items"],
        "is_refrigerated": False,
        "is_secure": True,
        "is_eye_level": True,
        "is_lower": False
    },
    "S2": {
        "type": "lower_shelf",
        "capacity": 25,
        "allowed_categories": ["grains", "bulk_items"],
        "is_refrigerated": False,
        "is_secure": False,
        "is_eye_level": False,
        "is_lower": True
    },
    "S4": {
        "type": "eye_level",
        "capacity": 15,
        "allowed_categories": ["breakfast", "general"],
        "is_refrigerated": False,
        "is_secure": False,
        "is_eye_level": True,
        "is_lower": False
    },
    "S5": {
        "type": "general_aisle",
        "capacity": 20,
        "allowed_categories": ["grains", "sauces"],
        "is_refrigerated": False,
        "is_secure": False,
        "is_eye_level": False,
        "is_lower": False
    },
    "R1": {
        "type": "refrigerator",
        "capacity": 20,
        "allowed_categories": ["dairy", "frozen"],
        "is_refrigerated": True,
        "is_secure": False,
        "is_eye_level": False,
        "is_lower": False
    },
    "H1": {
        "type": "hazardous",
        "capacity": 10,
        "allowed_categories": ["hazardous"],
        "is_refrigerated": False,
        "is_secure": True,
        "is_eye_level": False,
        "is_lower": False
    }
}

products = {
    "P1": {
        "name": "Milk",
        "weight": 5,
        "category": "dairy",
        "perishable": True,
        "high_demand": True,
        "heavy": False,
        "hazardous": False,
        "complementary": [],
        "valid_shelves": ["R1"],
        "promotional": False,
        "theft_risk": False
    },
    "P2": {
        "name": "Rice Bag",
        "weight": 10,
        "category": "grains",
        "perishable": False,
        "high_demand": False,
        "heavy": True,
        "hazardous": False,
        "complementary": ["P5", "P6"],
        "valid_shelves": ["S2", "S5"],
        "promotional": False,
        "theft_risk": False
    },
    "P3": {
        "name": "Frozen Nuggets",
        "weight": 5,
        "category": "frozen",
        "perishable": True,
        "high_demand": False,
        "heavy": False,
        "hazardous": False,
        "complementary": [],
        "valid_shelves": ["R1"],
        "promotional": False,
        "theft_risk": False
    },
    "P4": {
        "name": "Cereal",
        "weight": 3,
        "category": "breakfast",
        "perishable": False,
        "high_demand": True,
        "heavy": False,
        "hazardous": False,
        "complementary": [],
        "valid_shelves": ["S4"],
        "promotional": True,
        "theft_risk": False
    },
    "P5": {
        "name": "Pasta",
        "weight": 2,
        "category": "grains",
        "perishable": False,
        "high_demand": False,
        "heavy": False,
        "hazardous": False,
        "complementary": ["P6"],
        "valid_shelves": ["S2", "S5"],
        "promotional": False,
        "theft_risk": False
    },
    "P6": {
        "name": "Pasta Sauce",
        "weight": 3,
        "category": "sauces",
        "perishable": False,
        "high_demand": False,
        "heavy": False,
        "hazardous": False,
        "complementary": ["P5"],
        "valid_shelves": ["S5"],
        "promotional": False,
        "theft_risk": False
    },
    "P7": {
        "name": "Detergent",
        "weight": 4,
        "category": "hazardous",
        "perishable": False,
        "high_demand": False,
        "heavy": False,
        "hazardous": True,
        "complementary": [],
        "valid_shelves": ["H1"],
        "promotional": False,
        "theft_risk": False
    },
    "P8": {
        "name": "Glass Cleaner",
        "weight": 5,
        "category": "hazardous",
        "perishable": False,
        "high_demand": False,
        "heavy": False,
        "hazardous": True,
        "complementary": [],
        "valid_shelves": ["H1"],
        "promotional": False,
        "theft_risk": True
    },
    "P9": {
        "name": "Lays",
        "weight": 5,
        "category": "snacks",
        "perishable": True,
        "high_demand": True,
        "heavy": False,
        "hazardous": False,
        "complementary": [],
        "valid_shelves": ["S1"],
        "promotional": False,
        "theft_risk": False
    }
}

# ======================
# GENETIC ALGORITHM CORE
# ======================
def generate_population(pop_size):
    population = []
    for _ in range(pop_size):
        allocation = {shelf: [] for shelf in shelves}
        for pid, pdata in products.items():
            valid_shelves = [
                s for s in pdata['valid_shelves']
                if (sum(products[item]['weight'] for item in allocation[s]) + pdata['weight'] <= shelves[s]['capacity'])
            ]
            if valid_shelves:
                shelf = random.choice(valid_shelves)
            else:
                shelf = random.choice(pdata['valid_shelves'])
            allocation[shelf].append(pid)
        population.append(allocation)
    return population

def calculate_penalties(allocation):
    penalties = 0
    category_map = defaultdict(set)
    complementary_map = defaultdict(set)
    refrigerated_products = []
    
    # Shelf-level checks
    for shelf, items in allocation.items():
        total_weight = sum(products[pid]['weight'] for pid in items)
        
        # Constraint 1: Weight capacity
        if total_weight > shelves[shelf]['capacity']:
            penalties += (total_weight - shelves[shelf]['capacity']) * 10
            
        # Constraint 3: Category compatibility
        for pid in items:
            pdata = products[pid]
            if pdata['category'] not in shelves[shelf]['allowed_categories']:
                penalties += 20
            category_map[pdata['category']].add(shelf)
            if pdata['complementary']:
                complementary_map[tuple(pdata['complementary'])].add(shelf)
            if pdata['perishable']:
                refrigerated_products.append( (pid, shelf) )

    # Constraint 2: High-demand accessibility
    for pid, pdata in products.items():
        if pdata['high_demand']:
            current_shelf = next((s for s, items in allocation.items() if pid in items), None)
            if current_shelf:
                if not (shelves[current_shelf]['is_eye_level'] or shelves[current_shelf]['type'] == 'checkout_display'):
                    penalties += 15

    # Constraint 4: Perishable handling
    for pid, pdata in products.items():
        if pdata['perishable']:
            current_shelf = next((s for s, items in allocation.items() if pid in items), None)
            if current_shelf and not shelves[current_shelf]['is_refrigerated']:
                penalties += 25

    # Constraint 5: Hazardous separation
    for pid, pdata in products.items():
        if pdata['hazardous']:
            current_shelf = next((s for s, items in allocation.items() if pid in items), None)
            if current_shelf != 'H1':
                penalties += 30

    # Constraint 6: Cross-selling
    for group, shelves_in in complementary_map.items():
        if len(shelves_in) > 1:
            penalties += 20 * (len(shelves_in) - 1)

    # Constraint 7: Restocking efficiency
    for pid, pdata in products.items():
        if pdata['heavy']:
            current_shelf = next((s for s, items in allocation.items() if pid in items), None)
            if current_shelf and not shelves[current_shelf]['is_lower']:
                penalties += 15

    # Constraint 8: Refrigeration Efficiency
    refrigerated_shelves_used = list(set(shelf for (pid, shelf) in refrigerated_products))
    refrigerated_shelves_used.sort()
    if len(refrigerated_shelves_used) > 1:
        total_weight_per_shelf = {}
        for s in refrigerated_shelves_used:
            total_weight_per_shelf[s] = sum(products[pid]['weight'] for pid in allocation[s] if products[pid]['perishable'])
        for i in range(len(refrigerated_shelves_used)):
            current_shelf = refrigerated_shelves_used[i]
            current_capacity = shelves[current_shelf]['capacity'] - total_weight_per_shelf.get(current_shelf, 0)
            for j in range(i+1, len(refrigerated_shelves_used)):
                later_shelf = refrigerated_shelves_used[j]
                for pid in allocation[later_shelf]:
                    if products[pid]['perishable'] and products[pid]['weight'] <= current_capacity:
                        penalties += 20
                        current_capacity -= products[pid]['weight']

    # Constraint 9: Promotional visibility
    for pid, pdata in products.items():
        if pdata.get('promotional', False):
            current_shelf = next((s for s, items in allocation.items() if pid in items), None)
            if current_shelf:
                if not (shelves[current_shelf]['is_eye_level'] or shelves[current_shelf]['type'] == 'checkout_display'):
                    penalties += 15

    # Constraint 10: Theft Prevention
    for pid, pdata in products.items():
        if pdata.get('theft_risk', False):
            current_shelf = next((s for s, items in allocation.items() if pid in items), None)
            if current_shelf and not shelves[current_shelf]['is_secure']:
                penalties += 20

    # Constraint 3: Category segmentation
    for category, shelves_used in category_map.items():
        if len(shelves_used) > 1:
            penalties += 15 * (len(shelves_used) - 1)

    return penalties

def tournament_selection(population, tournament_size=3):
    selected = []
    for _ in range(2):
        contestants = random.sample(population, tournament_size)
        selected.append(min(contestants, key=lambda x: calculate_penalties(x)))
    return selected

def crossover(parent1, parent2):
    child = {shelf: [] for shelf in shelves}
    for shelf in shelves:
        child[shelf] = parent1[shelf].copy() if random.random() < 0.5 else parent2[shelf].copy()
    return repair(child)

def mutate(solution):
    shelves_list = list(shelves.keys())
    for _ in range(3):
        s1, s2 = random.sample(shelves_list, 2)
        items_s1 = [pid for pid in solution[s1] if s2 in products[pid]['valid_shelves']]
        items_s2 = [pid for pid in solution[s2] if s1 in products[pid]['valid_shelves']]
        
        if items_s1 and items_s2:
            item1 = random.choice(items_s1)
            item2 = random.choice(items_s2)
            if (sum(products[p]['weight'] for p in solution[s1] if p != item1) + products[item2]['weight'] <= shelves[s1]['capacity'] and
                sum(products[p]['weight'] for p in solution[s2] if p != item2) + products[item1]['weight'] <= shelves[s2]['capacity']):
                solution[s1].remove(item1)
                solution[s2].remove(item2)
                solution[s1].append(item2)
                solution[s2].append(item1)
    return repair(solution)

def repair(solution):
    for shelf in shelves:
        while sum(products[pid]['weight'] for pid in solution[shelf]) > shelves[shelf]['capacity']:
            items = sorted(solution[shelf], key=lambda x: products[x]['weight'], reverse=True)
            if not items:
                break
            item = items[0]
            valid_shelves = [
                s for s in products[item]['valid_shelves']
                if sum(products[p]['weight'] for p in solution[s]) + products[item]['weight'] <= shelves[s]['capacity']
            ]
            if valid_shelves:
                new_shelf = random.choice(valid_shelves)
                solution[shelf].remove(item)
                solution[new_shelf].append(item)
            else:
                break
    return solution

def genetic_algorithm(generations=200, pop_size=100):
    population = generate_population(pop_size)
    best_solution = min(population, key=lambda x: calculate_penalties(x))
    best_fitness = calculate_penalties(best_solution)
    
    for _ in range(generations):
        parents = tournament_selection(population)
        offspring = [crossover(*random.sample(parents, 2)) for _ in range(pop_size - len(parents))]
        population = parents + [mutate(child) for child in offspring]
        
        current_best = min(population, key=lambda x: calculate_penalties(x))
        current_fitness = calculate_penalties(current_best)
        if current_fitness < best_fitness:
            best_solution = current_best.copy()
            best_fitness = current_fitness
        if best_fitness == 0:
            break
    return best_solution

# ======================
# VISUALIZATION & EXPORT
# ======================

def save_to_excel(solution, filename="shelf_optimization.csv"):
    data = []
    for shelf, items in solution.items():
        for pid in items:
            pdata = products[pid]
            data.append({
                "Shelf": shelf,
                "Shelf Type": shelves[shelf]['type'],
                "Product": pdata['name'],
                "Weight (kg)": pdata['weight'],
                "Category": pdata['category'],
                "Perishable": "Yes" if pdata['perishable'] else "No",
                "High Demand": "Yes" if pdata['high_demand'] else "No",
                "Hazardous": "Yes" if pdata['hazardous'] else "No",
                "Promotional": "Yes" if pdata.get('promotional', False) else "No",
                "Theft Risk": "Yes" if pdata.get('theft_risk', False) else "No"
            })
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

# ======================
# EXECUTION
# ======================

if __name__ == "__main__":
    solution = genetic_algorithm(generations=200, pop_size=100)
    print(f"Optimal solution found with penalty score: {calculate_penalties(solution)}")
    save_to_excel(solution, "shelf_optimization.csv")
    print("Shelf allocation saved to 'shelf_optimization.csv'.")
