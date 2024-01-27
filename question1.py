import random
import matplotlib.pyplot as plt
import time
import numpy as np

# Genetic Algorithm Parameters
population_size = 200
mutation_probability = 0.25
crossover_probability = 0.85
generations = 100
string_length = 6 # Number of genes/bytes in each string
num_strings = 4 # Number of strings in each chromosome
fitness_function = "(3 * x1**2 * x3 * x4**3 ) + (2 * x2 * x3**3 * x4**2) + (4 * x1 * x2)" 
ranges = [(2, 5), (5, 10), (0, 6), (10, 15)]
chromesome_length = num_strings * string_length # Number of genes/bytes in each chromosome

# Function to generate initial generation
def generateInitialPopulation():
    # Generate a population of random binary numbers
    population = np.random.randint(0, 2, size=(population_size, chromesome_length))    
    return population

# Function to convert binary to decimal
def binary_to_decimal(binary):
    binary_string = ''.join(map(str, binary)) # Convert each element in the binary list to a string and then join them together.
    decimal_value = int(binary_string, 2) # Convert the binary string to a decimal value
    return decimal_value

# Evaluation: Function to decode and evaluate the fitness of each chromosome
def evalation(population_matrix):
    # Empty matrix to store the decoded values of each chromosome
    de_values = np.zeros((population_size, num_strings))

    for i in range(population_size): # For each chromosome
        for j in range(num_strings): # For each string in the chromosome
            start_index = j * string_length
            end_index = start_index + string_length
            
            lower = ranges[j][0] # The lower bound of the current string
            upper = ranges[j][1] # The upper bound of the current string

            binary = population_matrix[i, start_index:end_index] # The binary representation of the current string. Note: i is the index of the current row you are interested in
            decimal = binary_to_decimal(binary) # The decimal representation of the current string
            de_values[i, j] = lower + (decimal * (upper - lower)) / ((2 ** string_length) - 1)


    # Evaluate fitness of each chromosome using the fitness function
    fitness = np.zeros((population_size, 1)) # Create a matrix to store the fitness of each chromosome
    for i in range(population_size): 
        x1, x2, x3, x4 = de_values[i]
        fitness[i] = eval(fitness_function)
        
    return fitness

# Function for roulette wheel selection
def roulette_wheel_selection(fitness, population_matrix):
    # Make all fitness values positive
    if min(fitness) < 0:
        fitness = fitness + abs(min(fitness))
        
    # Calculate the total fitness of the population
    cumulative_fitness = np.cumsum(fitness)
    total_fitness = np.sum(fitness)

    # Generate random numbers between 0 and 1 for the roulette wheel
    random_numbers = np.random.random_sample(population_size)

    # Select the chromosomes that corresponds to the random number based on the cumulative fitness
    selected_chromosomes = []
    
    for number in random_numbers:
        random_number = number * total_fitness
        for i in range(population_size):
            if random_number < cumulative_fitness[i]:
                selected_chromosomes.append(i)
                break

    # Get the index of chromosome with the best fitness
    best_index = np.argmax(fitness)

    # Create a new population matrix with the selected chromosomes
    new_population_matrix = np.array(population_matrix[selected_chromosomes])

    return best_index, new_population_matrix


# Function for two-point crossover
def two_point_crossover(parent1, parent2, crossover_points):
    child1 = np.concatenate((parent1[:crossover_points[0]], parent2[crossover_points[0]:crossover_points[1]], parent1[crossover_points[1]:]))
    child2 = np.concatenate((parent2[:crossover_points[0]], parent1[crossover_points[0]:crossover_points[1]], parent2[crossover_points[1]:]))
    return child1, child2

# Function to perform crossover
def crossover(population_matrix):
    # Initialize a new matrix to store the children
    new_population_matrix = np.zeros_like(population_matrix)

    for i in range(population_size // 2): # For each chromosome pair
        # Select two random indices for crossover pair
        parent_indices = np.random.choice(population_size, size=2, replace=False)
        parent1, parent2 = population_matrix[parent_indices]

        if np.random.rand() < crossover_probability:     
            # Select two random crossover points       
            crossover_points = np.sort(np.random.choice(string_length, size=2, replace=False))

            for j in range(num_strings): # For each string in the chromosome
                start_index = j * string_length # The start index of the current string
                end_index = start_index + string_length

                parent1_string = parent1[start_index:end_index]
                parent2_string = parent2[start_index:end_index]

                child1, child2 = two_point_crossover(parent1_string, parent2_string, crossover_points)
                new_population_matrix[i * 2, start_index:end_index] = child1
                new_population_matrix[i * 2 + 1, start_index:end_index] = child2

        else:
            # Copy the parents into the new population matrix as children
            new_population_matrix[i * 2] = parent1
            new_population_matrix[i * 2 + 1] = parent2

    return new_population_matrix

# Function to perform mutation
def mutation(population_matrix):
    # Initialize a new matrix to store the children
    new_population_matrix = np.zeros_like(population_matrix)

    # Perform mutation for each chromosome
    for i in range(population_size // 2):

        # Select two random indices for mutation pair
        parent_indices = np.random.choice(population_size, size=2, replace=False)

        # Get the chromosomes for mutation
        parent1 = population_matrix[parent_indices[0]]
        parent2 = population_matrix[parent_indices[1]]

        new_population_matrix[i * 2] = parent1
        new_population_matrix[i * 2 + 1] = parent2

        # Check if mutation should be performed based on the mutation rate
        if np.random.rand() < mutation_probability:
            # Select a random mutation point
            mutation_point = np.random.randint(1, string_length)

            # Get the index of the mutation point for each string
            mutate_point = string_length - mutation_point

            # Perform mutation for each string
            for j in range(num_strings):
                # Perform mutation for the current string
                new_population_matrix[i * 2, mutate_point] = parent2[mutate_point]
                new_population_matrix[i * 2 + 1, mutate_point] = parent1[mutate_point]

                mutate_point += string_length

    return new_population_matrix

# Main Genetic Algorithm function
def main():
    start_time = time.time() # Start time of the algorithm
    
    # Initial Generation i.e generation = 0
    population_matrix = generateInitialPopulation()
    fitness = evalation(population_matrix)
    best_index, new_population_matrix = roulette_wheel_selection(fitness, population_matrix)

    best_solution = population_matrix[best_index] 
    best_fitness = fitness[best_index]
    
    best_fitness_per_generation = []
    generation_list = []

    for generation in range(1, generations + 1):
        # Update the population matrix
        population_matrix = new_population_matrix

        # Perform crossover
        population_matrix = crossover(population_matrix)

        # Perform mutation
        population_matrix = mutation(population_matrix)

        # Evaluate fitness
        fitness = evalation(population_matrix)

        # Selection
        best_index, new_population_matrix = roulette_wheel_selection(fitness, population_matrix)

        # Update best solution if a better solution is found
        if fitness[best_index] > best_fitness:
            best_solution = population_matrix[best_index]
            best_fitness = fitness[best_index]

        # Store data for plotting
        generation_list.append(generation)
        best_fitness_per_generation.append(best_fitness)

        # Print the best solution and fitness for each generation
        print(f"Generation {generation}: Best Solution: {best_solution}, Best Fitness: {best_fitness}")
    
    end_time = time.time() # End time of the algorithm
    computation_time = end_time - start_time # Computation time of the algorithm

    # Print the best solution found so far
    print("Best Solution: ", best_solution)
    print("Best Fitness: ", best_fitness)

    # Plotting
    plt.plot(generation_list, best_fitness_per_generation, marker='o', color='violet')
    plt.title('Best Fitness Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.grid()
    plt.savefig("question1.png")
    plt.show()

    print(f"\nComputation Time: {computation_time} seconds")


if __name__ == "__main__":
    main()
