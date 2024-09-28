import random
import numpy as np
import pandas as pd

# Constants
T = 11721
N_PERIODS = 8
STAND_SIZE = 40
STEP_SIZE = 5
MIN_AGE = 35
POPULATION_SIZE = 50
GENERATIONS = 100
MUTATION_RATE = 0.1
TOURNAMENT_SIZE = 3
NUM_ELITES = 1

# Import data
stands_data = pd.read_csv('../data/Hypothetica_stand_data.csv')
adjacency_matrix = pd.read_csv('../data/Hypothetica_adjacency_list_thinned.csv')
volume_per_acre = stands_data[[f'P{j + 1}_volume' for j in range(N_PERIODS)]].values
initial_ages = stands_data['Age'].values
ages = initial_ages.copy()
n_stands = len(stands_data)
volume_per_stand = volume_per_acre * STAND_SIZE

# Create adjacency set for faster lookups
adjacency_set = set()
for i, row in adjacency_matrix.iterrows():
    adjacency_set.add((row['stand'], row['adjacent_stand']))


class HarvestSchedule:
    def __init__(self):
        self.schedule = {period: [] for period in range(N_PERIODS)}
        self.volume_by_period = [0] * N_PERIODS
        self.fitness_score = None
        self.constraint_violations = []
        self.harvested_stands = set()

    def __str__(self):
        """Return a string representation of the harvest schedule."""
        result = ["Volume Harvested by Period:"]

        for period, volume in enumerate(self.volume_by_period):
            result.append(f"Period {period + 1}: {volume:.2f} cubic feet")

        result.append("\nHarvest Schedule:")
        for period, stands in self.schedule.items():
            if stands:
                stands_str = ", ".join(map(str, stands))
                result.append(f"Period {period + 1}: Stands {stands_str}")
            else:
                result.append(f"Period {period + 1}: No stands scheduled")

        return "\n".join(result)

    def check_constraints(self):
        """Check for constraint violations and return True if no violations found."""
        violations = []
        harvested_stands = set()

        for period, stands in self.schedule.items():
            for i, stand in enumerate(stands):
                # Single-harvest constraint
                if stand in harvested_stands:
                    violations.append(f"Stand {stand} is harvested more than once.")
                else:
                    harvested_stands.add(stand)

                # Age constraint
                age_in_period = ages[stand] + STEP_SIZE * period
                if age_in_period < MIN_AGE:
                    violations.append(f"Stand {stand} is too young in period {period} (age: {age_in_period}).")

                # Adjacency constraint
                for j in range(i + 1, len(stands)):
                    other_stand = stands[j]
                    if (stand, other_stand) in adjacency_set or (other_stand, stand) in adjacency_set:
                        violations.append(f"Stands {stand} and {other_stand} are adjacent in period {period}.")

        self.constraint_violations.extend(violations)
        return len(violations) == 0

    def add(self, stand, period):
        """ Add a stand to the schedule and update the harvested volume. """
        if stand not in self.schedule[period]:
            self.schedule[period].append(stand)
            if 0 <= stand < len(volume_per_stand) and 0 <= period < len(volume_per_stand[0]):
                self.volume_by_period[period] += volume_per_stand[stand][period]
            self.harvested_stands.add(stand)


# Fitness calculation
def calculate_fitness(schedule: HarvestSchedule) -> float:
    """Calculate the fitness score based on deviation from target volume and penalties for violations."""
    penalty = 0
    if not schedule.check_constraints():
        penalty += 10000  # Large penalty for infeasible solutions

    deviation_sum = sum((schedule.volume_by_period[period] - T) ** 2 for period in range(N_PERIODS))

    # Return a very low fitness score for infeasible schedules
    if penalty > 0:
        return 1 / (1 + deviation_sum + penalty)

    # Return fitness for feasible schedules
    return 1 / (1 + deviation_sum)


# Genetic Algorithm subroutines: initialization, crossover, mutation, and selection
# Greedy volume-based initialization
def initialize_population() -> list:
    population = []

    for _ in range(POPULATION_SIZE):
        schedule = HarvestSchedule()

        for period in range(N_PERIODS):
            total_volume = 0
            stands_in_period = set()

            while total_volume < T:
                stand = random.randint(0, n_stands - 1)

                if stand in schedule.harvested_stands:
                    continue

                age_in_period = ages[stand] + STEP_SIZE * period
                if age_in_period >= MIN_AGE:
                    schedule.add(stand, period)
                    stands_in_period.add(stand)
                    schedule.harvested_stands.add(stand)

                    total_volume += volume_per_stand[stand][period]

                if len(stands_in_period) >= n_stands:
                    break

        population.append(schedule)

    return population


# One-point crossover
def crossover(parent1: HarvestSchedule, parent2: HarvestSchedule) -> HarvestSchedule:
    """Perform one-point crossover to create a child schedule, ensuring no stand is reused."""
    child = HarvestSchedule()
    child.harvested_stands = set()

    crossover_point = random.randint(0, N_PERIODS - 1)

    # Copy from parent1 before the crossover point
    for period in range(crossover_point):
        for stand in parent1.schedule[period]:
            if stand not in child.harvested_stands:
                child.add(stand, period)

    # Copy from parent2 after the crossover point
    for period in range(crossover_point, N_PERIODS):
        for stand in parent2.schedule[period]:
            if stand not in child.harvested_stands:
                child.add(stand, period)

    return child


# Mutation: Swap only
def mutate(schedule: HarvestSchedule) -> HarvestSchedule:
    """Mutate the schedule by swapping stands between two random periods, ensuring no stand is reused."""
    mutated_schedule = HarvestSchedule()
    mutated_schedule.schedule = {period: list(stands) for period, stands in schedule.schedule.items()}
    mutated_schedule.volume_by_period = list(schedule.volume_by_period)
    mutated_schedule.harvested_stands = set(schedule.harvested_stands)

    period1, period2 = random.sample(range(N_PERIODS), 2)

    # Ensure the swap doesn't reuse a stand that has already been harvested
    if mutated_schedule.schedule[period1] and mutated_schedule.schedule[period2]:
        stand1 = random.choice(mutated_schedule.schedule[period1])
        stand2 = random.choice(mutated_schedule.schedule[period2])

        if stand2 not in mutated_schedule.harvested_stands and stand1 not in mutated_schedule.harvested_stands:
            # Swap stands
            mutated_schedule.schedule[period1].remove(stand1)
            mutated_schedule.schedule[period2].remove(stand2)
            mutated_schedule.schedule[period1].append(stand2)
            mutated_schedule.schedule[period2].append(stand1)

            # Update volumes
            mutated_schedule.volume_by_period[period1] = mutated_schedule.volume_by_period[period1] - \
                                                         volume_per_stand[stand1][period1] + volume_per_stand[stand2][
                                                             period1]
            mutated_schedule.volume_by_period[period2] = mutated_schedule.volume_by_period[period2] - \
                                                         volume_per_stand[stand2][period2] + volume_per_stand[stand1][
                                                             period2]

    return mutated_schedule


# Tournament selection
def select_parents(population: list) -> tuple:
    """Select two parents using tournament selection."""

    def tournament_selection():
        tournament = random.sample(population, TOURNAMENT_SIZE)
        return max(tournament, key=lambda schedule: schedule.fitness_score)

    parent1 = tournament_selection()
    parent2 = tournament_selection()
    while parent2 == parent1:
        parent2 = tournament_selection()

    return parent1, parent2


# Elitism
def apply_elitism(population: list, new_population: list, num_elites: int = 1):
    """Carry over the best individuals to the new population."""
    sorted_population = sorted(population, key=lambda schedule: schedule.fitness_score, reverse=True)
    for i in range(num_elites):
        new_population.append(sorted_population[i])


# Main genetic algorithm loop
def genetic_algorithm(target_volume: int) -> HarvestSchedule:
    """Run the genetic algorithm to find the best harvest schedule."""
    population = initialize_population()

    for generation in range(GENERATIONS):
        print(f"\nGeneration {generation + 1}")

        # Calculate fitness for each schedule
        for schedule in population:
            if schedule.fitness_score is None:  # Calculate fitness only if it's not already set
                schedule.fitness_score = calculate_fitness(schedule)

        new_population = []

        # Apply elitism
        apply_elitism(population, new_population, num_elites=NUM_ELITES)

        # Generate new population through crossover and mutation
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = select_parents(population)
            child = crossover(parent1, parent2)
            if random.random() < MUTATION_RATE:
                child = mutate(child)
            new_population.append(child)

        population = new_population

    # Ensure all fitness scores are calculated before selecting the best schedule
    for schedule in population:
        if schedule.fitness_score is None:
            schedule.fitness_score = calculate_fitness(schedule)

    # Return the best schedule
    return max(population, key=lambda schedule: schedule.fitness_score)


def main():
    best_schedule = genetic_algorithm(T)
    print("Best Harvest Schedule:")
    print(best_schedule)


if __name__ == "__main__":
    main()
