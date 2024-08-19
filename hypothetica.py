import pulp
import pandas as pd

# Load data from CSV
data = pd.read_csv('Hypothetica_stand_data.csv')
volume_per_acre = data[[f'P{j+1}_volume' for j in range(8)]].values
initial_ages = data['Age'].values
n_stands = len(data)
n_periods = 8
acres_per_stand = 40
T = 11721.4

# Load adjacency list
adjacency_list = pd.read_csv('Hypothetica_adjacency_list_thinned.csv')
adjacency_list.columns = adjacency_list.columns.str.strip()
adjacency_pairs = adjacency_list[['stand', 'adjacent_stand']].values - 1

# Define the problem
prob = pulp.LpProblem("Harvest_Plan", pulp.LpMinimize)

# Define decision variables
x = pulp.LpVariable.dicts("x", ((i, j) for i in range(n_stands) for j in range(n_periods)), cat='Binary')

# Objective function: Minimize the deviation from the target harvest volume
deviation = pulp.LpVariable.dicts("deviation", (j for j in range(n_periods)), lowBound=0)

# Objective: Minimize the sum of deviations
prob += pulp.lpSum(deviation[j] for j in range(n_periods))

# Constraints

# 1. Each stand can only be harvested once
for i in range(n_stands):
    prob += pulp.lpSum([x[(i, j)] for j in range(n_periods)]) == 1

# 2. Minimum harvest age constraint (with age adjustment)
for i in range(n_stands):
    for j in range(n_periods):
        current_age = initial_ages[i] + j * 5  # Adjust the age for the current period
        if current_age < 35:
            prob += x[(i, j)] == 0

# 3. Adjacency constraint
for pair in adjacency_pairs:
    i, k = pair  # Already adjusted for zero-based indexing
    for j in range(n_periods):
        prob += x[(i, j)] + x[(k, j)] <= 1

# 4. Harvest volume in each period and deviation constraint
for j in range(n_periods):
    harvest_volume = pulp.lpSum([acres_per_stand * volume_per_acre[i][j] * x[(i, j)] for i in range(n_stands)])
    prob += harvest_volume - deviation[j] <= T
    prob += harvest_volume + deviation[j] >= T

# Solve the problem
prob.solve()

# Output the solution
for i in range(n_stands):
    for j in range(n_periods):
        if pulp.value(x[(i, j)]) == 1:
            print(f"Stand {i+1} is harvested in period {j+1}")
