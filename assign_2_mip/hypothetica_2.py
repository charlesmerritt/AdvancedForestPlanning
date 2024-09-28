import pulp
import pandas as pd

# Load data from CSV
data = pd.read_csv('../data/Hypothetica_stand_data.csv')
volume_per_acre = data[[f'P{j+1}_volume' for j in range(8)]].values
initial_ages = data['Age'].values
n_stands = len(data)
n_periods = 8
acres_per_stand = 40
T = 11721

# Load adjacency list and strip any leading/trailing whitespace from the column names
adjacency_list = pd.read_csv('../data/Hypothetica_adjacency_list_thinned.csv')
adjacency_list.columns = adjacency_list.columns.str.strip()

# Use the corrected column names
adjacency_pairs = adjacency_list[['stand', 'adjacent_stand']].values - 1

# Define the problem
prob = pulp.LpProblem("Harvest_Plan", pulp.LpMinimize)

# Define binary decision variables for MIP: fully harvested (1) or not harvested (0)
x = pulp.LpVariable.dicts("x", ((i, j) for i in range(n_stands) for j in range(n_periods)), cat='Binary')

# Objective function: Minimize the deviation from the target harvest volume
deviation = pulp.LpVariable.dicts("deviation", (j for j in range(n_periods)), lowBound=0)

# Objective: Minimize the sum of deviations from the target harvest volume
prob += pulp.lpSum(deviation[j] for j in range(n_periods))

# Constraints

# 1. Each stand can be harvested at most once
for i in range(n_stands):
    prob += pulp.lpSum([x[(i, j)] for j in range(n_periods)]) <= 1

# 2. Minimum harvest age constraint (with age adjustment)
for i in range(n_stands):
    for j in range(n_periods):
        current_age = initial_ages[i] + j * 5  # Adjust the age for the current period
        if current_age < 35:
            prob += x[(i, j)] == 0  # Prevent harvesting if age is less than 35

# 3. Adjacency constraint: No two adjacent stands can be harvested in the same period
for pair in adjacency_pairs:
    i, k = pair
    for j in range(n_periods):
        prob += x[(i, j)] + x[(k, j)] <= 1

# 4. Harvest volume in each period and deviation constraint
for j in range(n_periods):
    harvest_volume = pulp.lpSum([acres_per_stand * volume_per_acre[i][j] * x[(i, j)] for i in range(n_stands)])
    prob += harvest_volume - deviation[j] <= T
    prob += harvest_volume + deviation[j] >= T

# Solve the problem
prob.solve()

# Collect the solution into a list
solution = []
for i in range(n_stands):
    for j in range(n_periods):
        if pulp.value(x[(i, j)]) == 1:
            solution.append([i + 1, j + 1])

# Convert the solution list to a DataFrame and save it to a CSV file
solution_df = pd.DataFrame(solution, columns=['Stand', 'Period'])
solution_df.to_csv('harvest_plan_solution_mip.csv', index=False)

print("Solution saved to 'harvest_plan_solution_mip.csv'")
