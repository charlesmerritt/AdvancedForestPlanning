import random
import pandas as pd
import numpy as np

# Constants
T = 11721
N_PERIODS = 8
STAND_SIZE = 40
STEP_SIZE = 5
MIN_AGE = 35
TABU_STATE = 100
MAX_ITER = 10000
NEIGHBORHOOD_SIZE = 1000
REVERSION_THRESHOLD = 1000

# Load data from CSV
stands_data = pd.read_csv('../data/Hypothetica_stand_data.csv')
volume_per_acre = stands_data[[f'P{j + 1}_volume' for j in range(N_PERIODS)]].values
initial_ages = stands_data['Age'].values
n_stands = len(stands_data)
volume_per_stand = volume_per_acre * STAND_SIZE

# Load adjacency list
adjacency_list = pd.read_csv('../data/Hypothetica_adjacency_list_thinned.csv')
adjacency_list.columns = adjacency_list.columns.str.strip()
adjacency_pairs = adjacency_list[['stand', 'adjacent_stand']].values - 1

# Create adjacency set for faster lookups
adjacency_set = set()
for i, row in adjacency_list.iterrows():
    adjacency_set.add((row['stand'] - 1, row['adjacent_stand'] - 1))


class HarvestSchedule:
    def __init__(self):
        self.schedule = {period: [] for period in range(N_PERIODS)}
        self.volume_by_period = [0] * N_PERIODS
        self.error = None
        self.constraint_violations = []
        self.harvested_stands = set()

    def __str__(self):
        result = ["Volume Harvested by Period:"]
        squared_deviations = []
        target_volume = T

        for period, volume in enumerate(self.volume_by_period):
            deviation = (volume - target_volume) ** 2
            squared_deviations.append(deviation)
            result.append(f"Period {period + 1}: {volume:.2f} cubic feet, Squared Deviation: {deviation:.2f}")

        result.append("\nHarvest Schedule:")
        for period, stands in self.schedule.items():
            if stands:
                stands_str = ", ".join(map(lambda x: str(x + 1), stands))
                result.append(f"Period {period + 1}: Stands {stands_str}")
            else:
                result.append(f"Period {period + 1}: No stands scheduled")

        return "\n".join(result)

    def check_constraints(self):
        violations = []
        harvested_stands = set()
        for period, stands in self.schedule.items():
            for i, stand in enumerate(stands):
                if stand in harvested_stands:
                    violations.append(f"Stand {stand} is harvested more than once.")
                else:
                    harvested_stands.add(stand)
                age_in_period = initial_ages[stand] + STEP_SIZE * period
                if age_in_period < MIN_AGE:
                    violations.append(f"Stand {stand} is too young in period {period} (age: {age_in_period}).")
                for j in range(i + 1, len(stands)):
                    other_stand = stands[j]
                    if (stand, other_stand) in adjacency_set or (other_stand, stand) in adjacency_set:
                        violations.append(f"Stands {stand} and {other_stand} are adjacent in period {period}.")
        self.constraint_violations = violations
        return len(violations) == 0

    def add(self, stand, period):
        if stand not in self.schedule[period]:
            self.schedule[period].append(stand)
            if 0 <= stand < len(volume_per_stand) and 0 <= period < len(volume_per_stand[0]):
                self.volume_by_period[period] += volume_per_stand[stand][period]
            self.harvested_stands.add(stand)

    def remove(self, stand, period):
        if stand in self.schedule[period]:
            self.schedule[period].remove(stand)
            self.volume_by_period[period] -= volume_per_stand[stand][period]
            if stand in self.harvested_stands:
                self.harvested_stands.remove(stand)

    def evaluate(self):
        target_volumes = [T] * N_PERIODS
        self.error = sum((volume - target) ** 2 for volume, target in zip(self.volume_by_period, target_volumes))
        return self.error


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


def tabu_search():
    current_solution = init_feasible_schedule()
    tabu_list = []
    best_solution = current_solution
    best_cost = current_solution.evaluate()
    iterations_since_improvement = 0

    for iteration in range(MAX_ITER):
        neighborhood = []
        for _ in range(NEIGHBORHOOD_SIZE):
            neighbor = HarvestSchedule()
            neighbor.schedule = {period: stands.copy() for period, stands in current_solution.schedule.items()}
            neighbor.volume_by_period = current_solution.volume_by_period.copy()
            neighbor.harvested_stands = current_solution.harvested_stands.copy()

            stand = random.randint(0, n_stands - 1)
            current_period = random.choice(list(range(N_PERIODS)))
            new_period = random.choice([p for p in range(N_PERIODS) if p != current_period])

            if stand in neighbor.schedule[current_period]:
                neighbor.remove(stand, current_period)
            neighbor.add(stand, new_period)

            if neighbor.check_constraints() and (stand, new_period) not in tabu_list:
                neighborhood.append((neighbor.evaluate(), neighbor, (stand, new_period)))

        if neighborhood:
            neighborhood.sort(key=lambda x: x[0])
            best_neighbor_cost, best_neighbor, move = neighborhood[0]

            if best_neighbor_cost < best_cost:
                best_solution = best_neighbor
                best_cost = best_neighbor_cost
                iterations_since_improvement = 0  # Reset counter
            else:
                iterations_since_improvement += 1

            tabu_list.append(move)
            if len(tabu_list) > TABU_STATE:
                tabu_list.pop(0)

            current_solution = best_neighbor

            # Revert to the best solution if no improvement within threshold
            if iterations_since_improvement >= REVERSION_THRESHOLD:
                print(f"Reverting to best solution at iteration {iteration}")
                current_solution = best_solution
                iterations_since_improvement = 0  # Reset after reversion

        else:
            print(f"Iteration {iteration}: No valid neighbors found; keeping current solution.")

        if iteration % 100 == 0:
            print(f"Iteration {iteration}: Best Error = {best_cost}")

    return best_solution


def main():
    best_schedule = tabu_search()
    print("Best Harvest Schedule Found:")
    print(best_schedule)
    print(f"Error: {best_schedule.error}")
    print(f"Constraint Violations: {best_schedule.constraint_violations}")


if __name__ == "__main__":
    main()
