import pandas as pd
# Stands can only be harvested once during the time horizon
# Adjacent stands are not harvested at the same time
# Stands are not harvested until they are at least 35 years old
# Stands that are scheduled for harvest should be completely harvested

# Set constants
T = 11721
N_PERIODS = 8
STAND_SIZE = 40
STEP_SIZE = 5

# Import data
stands_data = pd.read_csv('../data/Hypothetica_stand_data.csv')
adjacency_matrix = pd.read_csv('../data/Hypothetica_adjacency_list_thinned.csv')
volume_per_acre = stands_data[[f'P{j+1}_volume' for j in range(8)]].values
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
        self.schedule = []
        self.harvest_volume = 0
        self.volume_by_period = {}

    def add(self, stand, period):
        self.schedule.append((stand, period))
        if period not in self.volume_by_period:
            self.volume_by_period[period] = 0
        self.volume_by_period[period] += volume_per_stand[stand, period - 1]

    def remove(self, stand, period):
        self.schedule.remove((stand, period))
        self.volume_by_period[period] -= volume_per_stand[stand, period - 1]

    def clear(self):
        self.schedule = []
        self.harvest_volume = 0
        self.volume_by_period = {}

    def check_constraints(self):
        violations = []

        # Check if any stands are harvested more than once
        harvested_stands = set()
        for stand, period in self.schedule:
            if stand in harvested_stands:
                violations.append(f"Violation: Stand {stand} is harvested more than once.")
            harvested_stands.add(stand)

        # Check adjacency constraint using adjacency_set
        for i, (stand1, period1) in enumerate(self.schedule):
            for stand2, period2 in self.schedule[i + 1:]:
                if period1 == period2 and (stand1, stand2) in adjacency_set:
                    violations.append(
                        f"Violation: Stands {stand1} and {stand2} are adjacent and scheduled for the same period {period1}.")

        # Check age constraint
        for stand, period in self.schedule:
            if ages[stand] < 35:
                violations.append(f"Violation: Stand {stand} is too young to be harvested (age {ages[stand]}).")

        if violations:
            return violations
        else:
            return "No constraint violations."

    def target_met_in_all_periods(self, target):
        # Check if the target was met in all periods
        return all(volume >= target for volume in self.volume_by_period.values())

    def __str__(self):
        return str(self.schedule)


def binary_search(target_initial=T, min_gap=100):
    LB = 0
    UB = None  # Undefined upper bound to start
    target = target_initial
    x = target_initial
    best_solution = None

    # Schedule harvests greedily for this period while checking constraints
    while True:
        if UB is not None and UB - LB < min_gap:
            break

        schedule = HarvestSchedule()
        harvested_stands = set()
        period = 1

        while period <= N_PERIODS:
            schedule.harvest_volume = 0
            # Sort stands
            sorted_stands = sort_stands(period, criteria='Age')
            for stand in sorted_stands['Stand']-1:
                if stand in harvested_stands:
                    continue
                if ages[stand] >= 35:
                    adjacent_to_scheduled = False
                    if schedule is not None:
                        for s, p in schedule.schedule:
                            if p == period and (stand, s) in adjacency_set:
                                adjacent_to_scheduled = True
                                break
                    if not adjacent_to_scheduled:
                        schedule.add(stand, period)
                        harvested_stands.add(stand)
                        volume = volume_per_stand[stand, period - 1]
                        schedule.harvest_volume += volume
                        if schedule.harvest_volume > target:
                            break
            period += 1
            grow_stands()
            #sorted_stands = sort_stands(period, criteria='Age')

        # Check if the target was met in all periods
            if schedule.target_met_in_all_periods(target):
                best_solution = schedule
                LB = target
                if UB is None:
                    target += x
                    print(f"Target increased: {target}")
                elif UB - LB < min_gap:
                    break
                else:
                    target = target + (0.5 * (UB - LB))
                    print(f"Target adjusted upwards: {target}")
            else:
                UB = target
                if UB - LB < min_gap:
                    best_solution = schedule
                    break
                target = 0.5 * (LB + UB)
                print(f"Target adjusted downwards: {target}")

        return best_solution


def sort_stands(period, data=stands_data, criteria=None):
    if criteria is None:
        return "Please provide a sorting criteria."
    elif criteria == 'Age':
        data['Age'] = ages
        return data.sort_values(by='Age', ascending=False)
    elif criteria == 'Highest Volume':
        volume_column = f'P{period}_volume'
        return data.sort_values(by=volume_column, ascending=False)
    elif criteria == 'Lowest Volume':
        volume_column = f'P{period}_volume'
        return data.sort_values(by=volume_column, ascending=True)
    elif criteria == 'Slowest Growth':
        return "Not implemented."
    elif criteria == 'Value Ratio':
        return "Not implemented."

def grow_stands():
    # Grow stands by step size
    for i in range(n_stands):
        ages[i] += STEP_SIZE

def print_harvest_stands(schedule):
    for stand, period in schedule.schedule:
        volume = volume_per_stand[stand, period - 1]
        print(f"Harvesting Stand {stand+1} in Period {period} with volume {volume}.")

def print_volume_by_period(schedule):
    print("Harvest Volume by Period:")
    for period, volume in schedule.volume_by_period.items():
        print(f"Period {period}: Total Harvest Volume = {volume}")

def main():
    schedule = binary_search(T)
    if schedule is None:
        print("No solution found.")
    else:
        print_harvest_stands(schedule)
        print_volume_by_period(schedule)

if __name__ == "__main__":
    main()

