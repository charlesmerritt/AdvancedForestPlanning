import random
import pandas as pd
import numpy as np

# Constants
T = 11721
N_PERIODS = 8
STAND_SIZE = 40
STEP_SIZE = 5
MIN_AGE = 35

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
                # Adjust stand numbering by adding 1
                stands_str = ", ".join(map(lambda x: str(x + 1), stands))
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

    def remove(self, stand, period):
        """ Remove a stand from the schedule and update the harvested volume. """
        if stand in self.schedule[period]:
            self.schedule[period].remove(stand)
            self.volume_by_period[period] -= volume_per_stand[stand][period]
            if stand in self.harvested_stands:
                self.harvested_stands.remove(stand)

    def evaluate(self):
        """ Evaluate the fitness score based on the squared deviations from the target volume (T). """
        target_volumes = [T] * N_PERIODS
        self.fitness_score = sum((volume - target) ** 2 for volume, target in zip(self.volume_by_period, target_volumes))
        return self.fitness_score


def init_feasible_schedule():
    """
    Initialize a feasible harvest schedule by randomly selecting stands to harvest in each period.
    """
    schedule = HarvestSchedule()
    for period in range(N_PERIODS):
        stands = random.sample(range(n_stands), random.randint(0, n_stands // 4))  # Random initial sample of stands
        for stand in stands:
            schedule.add(stand, period)
        if not schedule.check_constraints():
            for stand in stands:  # Remove invalid stands if any violations occur
                schedule.remove(stand, period)
    schedule.evaluate()
    return schedule

def simulated_annealing(initial_temperature, cooling_rate):
    """
    Simulated Annealing to minimize the fitness score (squared deviations) while respecting constraints.
    """
    temperature = initial_temperature
    current_solution = init_feasible_schedule()
    best_solution = current_solution

    while temperature > 1:
        # Create a new solution by making a small change to the current one
        new_solution = HarvestSchedule()
        new_solution.schedule = {period: stands.copy() for period, stands in current_solution.schedule.items()}
        new_solution.volume_by_period = current_solution.volume_by_period.copy()

        # Randomly select a period and stand to modify
        period = random.randint(0, N_PERIODS - 1)
        stand = random.randint(0, n_stands - 1)

        if stand in new_solution.schedule[period]:
            new_solution.remove(stand, period)
        else:
            new_solution.add(stand, period)

        # Check the constraints and evaluate the new solution
        if new_solution.check_constraints():
            new_solution.evaluate()

            # Decide whether to accept the new solution
            if (new_solution.fitness_score < current_solution.fitness_score or
                    random.random() < np.exp((current_solution.fitness_score - new_solution.fitness_score) / temperature)):
                current_solution = new_solution

            # Update the best solution found so far
            if current_solution.fitness_score < best_solution.fitness_score:
                best_solution = current_solution

        # Decrease temperature
        temperature *= (1 - cooling_rate)

    return best_solution

def main():
    initial_temperature = 100000000
    cooling_rate = 0.000003
    best_schedule = simulated_annealing(initial_temperature, cooling_rate)
    print(best_schedule)
    print(f"Fitness Score: {best_schedule.fitness_score}")
    print(f"Constraint Violations: {best_schedule.constraint_violations}")

if __name__ == '__main__':
    main()
