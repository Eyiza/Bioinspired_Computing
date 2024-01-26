import random
import matplotlib.pyplot as plt
import time

# Genetic Algorithm Parameters
population_size = 200
mutation_probability = 0.25
crossover_probability = 0.85
generations = 100
num_strings = 4

# Function to generate a random individual with decimal values
def generate_individual():
    return [random.uniform(0, 1) for _ in range(num_strings)]

# Function to decode chromosome to real values within specified ranges
def decode_chromosome(chromosome, ranges):
    decoded = [lower + value * (upper - lower) for value, (lower, upper) in zip(chromosome, ranges)]
    return decoded

# Function to evaluate the fitness of an individual
def fitness(x, y, z, k):
    return 3 * x**2 * y * z**3 + 2 * y * z**3 * k**2 + 4 * x * y

# Function for two-point crossover
def crossover(parent1, parent2):
    crossover_point = random.randint(1, num_strings - 2)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# Function to perform mutation
def mutate(individual):
    for i in range(num_strings):
        if random.uniform(0, 1) < mutation_probability:
            individual[i] = random.uniform(0, 1)
    return individual

# Function for roulette wheel selection
def roulette_wheel_selection(population, fitness_values):
    total_fitness = sum(fitness_values)
    probabilities = [fit / total_fitness for fit in fitness_values]
    selected = random.choices(population, weights=probabilities)
    return selected[0]

# Rank-based selection for elitism
def rank_selection(population, fitness_values, elitism_count):
    ranked_population = [x for _, x in sorted(zip(fitness_values, population), key=lambda pair: pair[0], reverse=True)]
    elite = ranked_population[:elitism_count]
    return elite

# Main Genetic Algorithm function
def genetic_algorithm(population_size, num_strings, generations):
    ranges = [(2, 5), (5, 10), (0, 6), (10, 15)]
    population = [generate_individual() for _ in range(population_size)]
    best_fitness_per_generation = []

    for generation in range(generations):
        decoded_population = [decode_chromosome(individual, ranges) for individual in population]
        fitness_values = [fitness(*decoded) for decoded in decoded_population]

        best_individual = population[fitness_values.index(max(fitness_values))]
        best_fitness_per_generation.append(max(fitness_values))

        print(f"Generation {generation + 1}: Best Fitness - {max(fitness_values)}")

        elitism_count = 20
        elite = rank_selection(population, fitness_values, elitism_count)
        new_population = elite

        while len(new_population) < population_size:
            parent1 = roulette_wheel_selection(population, fitness_values)
            parent2 = roulette_wheel_selection(population, fitness_values)

            if random.uniform(0, 1) < crossover_probability:
                offspring1, offspring2 = crossover(parent1, parent2)
                offspring1 = mutate(offspring1)
                offspring2 = mutate(offspring2)
                new_population.extend([offspring1, offspring2])
            else:
                new_population.extend([mutate(parent1), mutate(parent2)])

        population = new_population

    return best_individual, best_fitness_per_generation

if __name__ == "__main__":
    start_time = time.time()

    best_solution_with_elitism, best_fitness_per_generation_with_elitism = genetic_algorithm(population_size, num_strings, generations)

    end_time = time.time()
    computation_time_with_elitism = end_time - start_time

    print("\nBest Solution with Elitism:", best_solution_with_elitism)
    print("Best Fitness with Elitism:", fitness(*decode_chromosome(best_solution_with_elitism, [(2, 5), (5, 10), (0, 6), (10, 15)])))
    
    # Plotting
    plt.plot(range(1, generations + 1), best_fitness_per_generation_with_elitism, marker='o', label='With Elitism')
    plt.title('Generation Number vs Best Fitness')
    plt.xlabel('Generation Number')
    plt.ylabel('Best Fitness')
    
    # Run the genetic algorithm without elitism for comparison
    _, best_fitness_per_generation_without_elitism = genetic_algorithm(population_size, num_strings, generations)
    plt.plot(range(1, generations + 1), best_fitness_per_generation_without_elitism, marker='o', label='Without Elitism')
    
    plt.legend()
    plt.show()

    print(f"\nComputation Time with Elitism: {computation_time_with_elitism} seconds")
