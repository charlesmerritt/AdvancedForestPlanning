import pandas as pd
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD, LpStatus

# Import adjacency list and stand data
adjacency = pd.read_csv('../data/Hypothetica_adjacency_list_thinned.csv')
stands = pd.read_csv('../data/Hypothetica_stand_data.csv')
stands_volume = pd.read_csv('../data/Hypothetica_stand_data_hv.csv')

# Ensure Stand IDs are integers
stands_volume['Stand'] = stands_volume['Stand'].astype(int)

# Create an LP minimization problem
prob = LpProblem("The_Hypothetica_Problem", LpMinimize)

# Goal is to minimize the squared deviation from the target for each period
target = 11721

# Create a dictionary to hold all decision variables
variables = {}

# Loop through stands and periods
for stand in range(1, 115):
    for period in range(1, 9):
        key = f"stand_{stand}period_{period}"
        variables[key] = LpVariable(key, lowBound=0, upBound=40)

# ideally: s1p1 + s2p1 + s3p1 + ... + s114p1 = 11721

# Create binary indicator variables for each stand and period
binary_variables = {}
for stand in range(1, 115):
    for period in range(1, 9):
        binary_variables[f"y_stand_{stand}period_{period}"] = LpVariable(f"y_stand_{stand}period_{period}", cat='Binary')

# Constraints
# Link continuous and binary variables
for stand in range(1, 115):
    for period in range(1, 9):
        prob += variables[f"stand_{stand}period_{period}"] <= 40 * binary_variables[f"y_stand_{stand}period_{period}"]

# Can't harvest stands that are less than 35 years old based on stand data (stands age 5 years per period)
for index, row in stands_volume.iterrows():
    stand = int(row['Stand'])
    age = row['Age']
    for period in range(1, 9):
        # Calculate the age of the stand at the beginning of the period
        period_age = age + (period - 1) * 5
        if period_age < 35:
            prob += variables[f"stand_{stand}period_{period}"] == 0
        # Stand starts at less than 35 years old but becomes mature by the end of the time horizon
        elif period_age >= 35:
            prob += variables[f"stand_{stand}period_{period}"] <= 40

# Enforce mutual exclusivity for adjacent stands
for index, row in adjacency.iterrows():
    stand1 = row['stand']
    stand2 = row['adjacent_stand']
    for period in range(1, 9):
        prob += binary_variables[f"y_stand_{stand1}period_{period}"] + binary_variables[f"y_stand_{stand2}period_{period}"] <= 1

# Can only harvest each stand once across the entire time horizon
for stand in range(1, 115):
    prob += lpSum([binary_variables[f"y_stand_{stand}period_{period}"] for period in range(1, 9)]) <= 1

deviation_variables = []

for period in range(1, 9):
    harvested_volume = lpSum([
        variables[f"stand_{stand}period_{period}"] *
        stands_volume.loc[stands_volume['Stand'] == stand, f'P{period}_volume'].values[0]
        for stand in range(1, 115)
    ])

    deviation = LpVariable(f"deviation_period_{period}", lowBound=0)

    # Deviation constraints (absolute deviation)
    prob += deviation >= harvested_volume - target
    prob += deviation >= target - harvested_volume

    deviation_variables.append(deviation)

# Objective: Minimize the sum of absolute deviations
prob += lpSum(deviation_variables)

# Solve the problem
status = prob.solve(PULP_CBC_CMD())

# Output results
if LpStatus[status] == 'Optimal':
    print("Optimal solution found.")

    # Iterate over periods
    for period in range(1, 9):
        # Iterate over stands
        for stand in range(1, 115):
            var_name = f"stand_{stand}period_{period}"
            if variables[var_name].varValue > 0:
                # Calculate the harvested volume in board feet for this stand and period
                harvested_volume = (
                        variables[var_name].varValue *
                        stands_volume.loc[stands_volume['Stand'] == stand, f'P{period}_volume'].values[0]
                )

                # Print the result in board feet
                print(f"{var_name}: {harvested_volume} board feet")
else:
    print("No optimal solution found.")