import random
import matplotlib.pyplot as plt
import time

# Genetic Algorithm Parameters
population_size = 200
mutation_probability = 0.25
crossover_probability = 0.85
generations = 100
string_length = 6
num_strings = 4

# Function to generate a random binary string
def generate_individual(length):
    return [random.choice([0, 1]) for _ in range(length)]

# Function to decode binary string to real values within specified ranges
def decode_chromosome(chromosome, ranges):
    decoded = []
    for i, (lower, upper) in enumerate(ranges):
        value = int("".join(map(str, chromosome[i * string_length:(i + 1) * string_length])), 2)
        decoded_value = lower + value * (upper - lower) / (2 ** string_length - 1)
        decoded.append(decoded_value)
    return decoded

# Function to evaluate the fitness of an individual
def fitness(x, y, z, k):
    return 3 * x**2 * y * z**3 + 2 * y * z**3 * k**2 + 4 * x * y

# Function for two-point crossover
def crossover(parent1, parent2):
    crossover_points = sorted(random.sample(range(len(parent1)), 2))
    child1 = parent1[:crossover_points[0]] + parent2[crossover_points[0]:crossover_points[1]] + parent1[crossover_points[1]:]
    child2 = parent2[:crossover_points[0]] + parent1[crossover_points[0]:crossover_points[1]] + parent2[crossover_points[1]:]
    return child1, child2

# Function to perform mutation
def mutate(individual):
    for i in range(len(individual)):
        if random.uniform(0, 1) < mutation_probability:
            individual[i] = 1 - individual[i]
    return individual

# Function for roulette wheel selection
def roulette_wheel_selection(population, fitness_values):
    total_fitness = sum(fitness_values)
    probabilities = [fit / total_fitness for fit in fitness_values]
    selected = random.choices(population, weights=probabilities)
    return selected[0]

# Main Genetic Algorithm function
def genetic_algorithm(population_size, num_strings, string_length, generations):
    ranges = [(2, 5), (5, 10), (0, 6), (10, 15)]
    population = [generate_individual(num_strings * string_length) for _ in range(population_size)]
    best_fitness_per_generation = []

    for generation in range(generations):
        decoded_population = [decode_chromosome(individual, ranges) for individual in population]
        fitness_values = [fitness(*decoded) for decoded in decoded_population]

        best_individual = population[fitness_values.index(max(fitness_values))]
        best_fitness_per_generation.append(max(fitness_values))

        print(f"Generation {generation + 1}: Best Fitness - {max(fitness_values)}")

        new_population = [best_individual]  # Keep the best individual

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

    best_solution, best_fitness_per_generation = genetic_algorithm(population_size, num_strings, string_length, generations)

    end_time = time.time()
    computation_time = end_time - start_time

    print("\nBest Solution:", best_solution)
    print("Best Fitness:", fitness(*decode_chromosome(best_solution, [(2, 5), (5, 10), (0, 6), (10, 15)])))
    
    # Plotting
    plt.plot(range(1, generations + 1), best_fitness_per_generation, marker='o')
    plt.title('Generation Number vs Best Fitness')
    plt.xlabel('Generation Number')
    plt.ylabel('Best Fitness')
    plt.show()

    print(f"\nComputation Time: {computation_time} seconds")
