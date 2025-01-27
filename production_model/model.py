import math
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from calendar import monthrange
import glob
from datetime import datetime

"""
The initial step in developing a production model is gathering the meteorological data and 
solar path information for the specific location of interest. The appendix includes a user 
manual for the production model. This section, however, focuses on the mathematical foundation 
for calculating the output of a solar installation. 

To estimate the yield of a 1-hectare project using the production model, we will employ four 
different models. This approach helps reduce uncertainty. The first model uses the coefficients 
of the linear regres- sion model. The second relies on the theoretical equation of the module. 
The third represents a best-case scenario considering new improvements, and the last model 
represents a worst-case scenario.

To run code one must first download from web sun path and irradiance data. The data can be found on 
websites like GLOBAL SOLAR ATLAS and SunEarthTools.com
"""

# ---------------------------------------------------------------------
# 1- The values below define tolerances, angles, and spacing used in later calculations.
# ---------------------------------------------------------------------

# Values for tolerance
tolerance_values_EL = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Fixed azimuth reference angle (south facing PV installation)
reference_angle = 180

# Values for tolerance
tolerance_values = [0, 5.7, 11.31, 16.70, 21.8, 26.57, 30.96, 34.99, 38.66, 41.99, 45]

# Space between reflector and PV panel
Az = [0.00,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10]


# ---------------------------------------------------------------------
# 2- processes irradiance data and generates 20-minute interval files for DNI and DHI.
# ---------------------------------------------------------------------

# Values to be modified depending on location
DHI = 604
DNI = 2103
ratio = DHI / DNI

# Path to the CSV file
file_path = '......./GSA_Report_Aznalcázar.csv'

# Read the CSV file, using the first row as headers and first column as the time of day
irradiance_data = pd.read_csv(file_path, header=0)

# Ensure the first column is treated as the time of day index
irradiance_data.set_index(irradiance_data.columns[0], inplace=True)

# Create a new DataFrame for 20-minute intervals
time_steps = [f"{h:02d}:{m:02d}:00" for h in range(24) for m in [0, 20, 40]]
dni_20min = pd.DataFrame(index=time_steps, columns=irradiance_data.columns)
dhi_20min = pd.DataFrame(index=time_steps, columns=irradiance_data.columns)

for hour in range(24):
    for minute in [0, 20, 40]:
        time_20min = f"{hour:02d}:{minute:02d}:00"
        if minute == 0:
            dni_20min.loc[time_20min] = irradiance_data.iloc[hour]
            dhi_20min.loc[time_20min] = irradiance_data.iloc[hour] * ratio
        elif minute == 20:
            next_hour = (hour + 1) % 24
            dni_20min.loc[time_20min] = irradiance_data.iloc[hour] + (irradiance_data.iloc[next_hour] - irradiance_data.iloc[hour]) / 3
            dhi_20min.loc[time_20min] = irradiance_data.iloc[hour] * ratio + (irradiance_data.iloc[next_hour] - irradiance_data.iloc[hour]) * ratio / 3
        elif minute == 40:
            next_hour = (hour + 1) % 24
            dni_20min.loc[time_20min] = irradiance_data.iloc[hour] + (irradiance_data.iloc[next_hour] - irradiance_data.iloc[hour]) * 2 / 3
            dhi_20min.loc[time_20min] = irradiance_data.iloc[hour] * ratio + (irradiance_data.iloc[next_hour] - irradiance_data.iloc[hour]) * ratio * 2 / 3

# Save the DNI and DHI matrices to new CSV files
dni_20min.to_csv('......./Production Model/DNI_Matrix_20min.csv', index_label='Time')
dhi_20min.to_csv('......./Production Model/DHI_Matrix_20min.csv', index_label='Time')

print("DNI and DHI matrices with 20-minute intervals have been saved.")

# ---------------------------------------------------------------------
# 3- process sun path data from the CSV file and compute average monthly elevation/azimuth angles.
# ---------------------------------------------------------------------

# Load the data from the CSV file
file_path = '......./SunEarthTools_AnnualSunPath_2024_1719555204504.csv'
data = pd.read_csv(file_path, delimiter=';')

# Rename the first column for clarity
data.rename(columns={data.columns[0]: 'Date'}, inplace=True)

# Drop any columns that are entirely NaN or irrelevant
data.drop(columns=['Unnamed: 145'], inplace=True)

# Extract month and day from the date column
data['Date'] = pd.to_datetime(data['Date'])
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day

# Separate elevation and azimuth angles
elevation_columns = [col for col in data.columns if col.startswith('E')]
azimuth_columns = [col for col in data.columns if col.startswith('A')]

# Melt the dataframe to have a single column for elevation and azimuth angles
elevation_data = data.melt(id_vars=['Date', 'Month', 'Day'], value_vars=elevation_columns, var_name='Time', value_name='Elevation')
azimuth_data = data.melt(id_vars=['Date', 'Month', 'Day'], value_vars=azimuth_columns, var_name='Time', value_name='Azimuth')

# Replace non-numeric values with 0
elevation_data['Elevation'] = pd.to_numeric(elevation_data['Elevation'].replace('--', 0), errors='coerce').fillna(0)
azimuth_data['Azimuth'] = pd.to_numeric(azimuth_data['Azimuth'].replace('--', 0), errors='coerce').fillna(0)

# Group by month and time to calculate the average angles
elevation_avg = elevation_data.groupby(['Month', 'Time'])['Elevation'].mean().unstack(level=0)
azimuth_avg = azimuth_data.groupby(['Month', 'Time'])['Azimuth'].mean().unstack(level=0)

# Adjust azimuth angles to ensure they do not decrease
azimuth_avg = azimuth_avg.apply(lambda x: x.where(x >= x.shift(fill_value=0), 0))

# Replace numeric months with their abbreviations
month_mapping = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
elevation_avg.columns = elevation_avg.columns.map(month_mapping)
azimuth_avg.columns = azimuth_avg.columns.map(month_mapping)

# Remove 'E ' and 'A ' from the index of the DataFrames
elevation_avg.index = elevation_avg.index.str.replace('E ', '')
azimuth_avg.index = azimuth_avg.index.str.replace('A ', '')

# Save the results to CSV files
elevation_avg.to_csv('......./Average_Elevation_Angles.csv')
azimuth_avg.to_csv('......./Average_Azimuth_Angles.csv')

print("Average Elevation Angles by Month and Time saved to 'Average_Elevation_Angles.csv'")
print("Average Azimuth Angles by Month and Time saved to 'Average_Azimuth_Angles.csv'")

# ---------------------------------------------------------------------
# 4- apply a conditional cosine transform to the azimuth angles and save the results.
# ---------------------------------------------------------------------

# Apply the cosine function to the azimuth angles
def conditional_cosine(azimuth_angle):
    if 60 <= azimuth_angle <= 300:
        return abs(math.cos(math.radians(azimuth_angle % 180)))
    else:
        return 0

azimuth_cos = azimuth_avg.applymap(conditional_cosine)

# Save the resulting DataFrame to a new CSV file
azimuth_cos.to_csv('......./Cos_Azimuth_Angles.csv')

print("Cosine of Average Azimuth Angles by Month and Time saved to 'Cos_Azimuth_Angles.csv'")

# ---------------------------------------------------------------------
# 5- define efficiency functions for azimuth and elevation based on reference angles and tolerances.
# ---------------------------------------------------------------------

# DEFINE EFFICIENCY FUNCTIONS
# 1- Azimuth efficiency function

def azimuth_efficiency(azimuth_angle, reference_angle, tolerance):
    if reference_angle - azimuth_angle >= 60 or reference_angle + azimuth_angle >= 300:
        if reference_angle - tolerance <= azimuth_angle <= reference_angle + tolerance:
            return 1
        elif reference_angle - tolerance - 24 <= azimuth_angle <= reference_angle + tolerance + 24:
            return 0.9
        elif reference_angle - tolerance - 33 <= azimuth_angle <= reference_angle + tolerance + 33:
            return 0.79
        elif reference_angle - tolerance - 41 <= azimuth_angle <= reference_angle + tolerance + 41:
            return 0.76
        elif reference_angle - tolerance - 47 <= azimuth_angle <= reference_angle + tolerance + 47:
            return 0.6
        elif reference_angle - tolerance - 52 <= azimuth_angle <= reference_angle + tolerance + 52:
            return 0.46
        elif reference_angle - tolerance - 56 <= azimuth_angle <= reference_angle + tolerance + 56:
            return 0.33
        elif reference_angle - tolerance - 62 <= azimuth_angle <= reference_angle + tolerance + 62:
            return 0.26
        else:
            return 0
    else:
        return 0

# 2- Elevation efficiency function

# Find the shortest day of the year (around December 21st)
shortest_day_data = elevation_data[(elevation_data['Month'] == 12) & (elevation_data['Day'] == 21)]
# Find the highest elevation angle on the shortest day
max_elevation_angle = shortest_day_data['Elevation'].max()

long_day_data = elevation_data[(elevation_data['Month'] == 7) & (elevation_data['Day'] == 21)]
# Find the highest elevation angle on the shortest day
max_elevation_angle_long_day = long_day_data['Elevation'].max()

# Fixed elevation reference angle (highest angle on the shortest day)
reference_angle_EL = max_elevation_angle + 30
max_el = max_elevation_angle_long_day

def elevation_efficiency(elevation_angle, reference_angle_EL, tolerance_EL):
    if elevation_angle >= max_elevation_angle_long_day + tolerance_EL:
        return 0.25
    elif elevation_angle >= reference_angle_EL + tolerance_EL:
        return 1
    elif elevation_angle >= reference_angle_EL + tolerance_EL - 1:
        return 0.99
    elif elevation_angle >= reference_angle_EL + tolerance_EL - 2:
        return 0.99
    elif elevation_angle >= reference_angle_EL + tolerance_EL - 3:
        return 0.99
    elif elevation_angle >= reference_angle_EL + tolerance_EL - 4:
        return 0.95
    elif elevation_angle >= reference_angle_EL + tolerance_EL - 5:
        return 0.7
    elif elevation_angle >= reference_angle_EL + tolerance_EL - 6:
        return 0.58
    elif elevation_angle >= reference_angle_EL + tolerance_EL - 7:
        return 0.5
    elif elevation_angle >= reference_angle_EL + tolerance_EL - 8:
        return 0.41
    elif elevation_angle >= reference_angle_EL + tolerance_EL - 9:
        return 0.31
    elif elevation_angle >= reference_angle_EL + tolerance_EL - 10:
        return 0.25
    else:
        return 0.25

# ---------------------------------------------------------------------
# 6- create efficiency matrices for each tolerance and save them as CSV files.
# ---------------------------------------------------------------------
    
# Create efficiency matrices and plots for each tolerance value using data
for tolerance in tolerance_values:
    efficiency_matrix = azimuth_avg.applymap(lambda azimuth_angle: azimuth_efficiency(azimuth_angle, reference_angle, tolerance))
    efficiency_matrix.index = efficiency_matrix.index.str.replace('A ', '')  # Remove 'A ' from index
    efficiency_matrix.to_csv(f'......./Efficiency_Matrix_Azimuth_Tolerance_{tolerance}.csv')

# Create efficiency matrices for each tolerance value (elevation) and save them to CSV
for tolerance_EL in tolerance_values_EL:
    efficiency_matrix = elevation_avg.applymap(lambda elevation_angle: elevation_efficiency(elevation_angle, reference_angle_EL, tolerance_EL))
    efficiency_matrix.index = efficiency_matrix.index.str.replace('E ', '')  # Remove 'E ' from index
    efficiency_matrix.to_csv(f'......./Efficiency_Matrix_Elevation_Tolerance_{tolerance_EL}.csv')

print("Efficiency matrices for azimuth and elevation saved to CSV files.")

# ---------------------------------------------------------------------
# 7- elevation efficiency matrices for given tolerances.
# ---------------------------------------------------------------------

def multiply_matrices(tolerance_azimuth, tolerance_elevation):
    # Load the data from the CSV files
    df_azimuth = pd.read_csv(f'......./Efficiency_Matrix_Azimuth_Tolerance_{tolerance_azimuth}.csv', index_col=0)
    df_elevation = pd.read_csv(f'......./Efficiency_Matrix_Elevation_Tolerance_{tolerance_elevation}.csv', index_col=0)
    
    # Multiply the dataframes element-wise
    df_result = df_azimuth.mul(df_elevation)

    return df_result

# Iterate over all combinations of tolerance values
for tolerance_azimuth in tolerance_values:
    for tolerance_elevation in tolerance_values_EL:
        try:
            result = multiply_matrices(tolerance_azimuth, tolerance_elevation)
            # Save the result to a CSV file
            result.to_csv(f'......./Production Model/Result_Azimuth_{tolerance_azimuth}_Elevation_{tolerance_elevation}.csv')
        except FileNotFoundError as e:
            print(f"File not found for tolerance_azimuth={tolerance_azimuth}, tolerance_elevation={tolerance_elevation}: {e}")
        except Exception as e:
            print(f"An error occurred for tolerance_azimuth={tolerance_azimuth}, tolerance_elevation={tolerance_elevation}: {e}")

print("All results have been processed and saved.")

# ---------------------------------------------------------------------
# 8- compute coverage ratios based on geometric considerations.
# ---------------------------------------------------------------------

def calculate_coverage_ratio(reference_angle_EL):
    coverage_ratios_and_D_values = {}
    max_elevation_rad = math.radians(reference_angle_EL)
    
    for j in range(len(tolerance_values_EL)):
        D = math.cos(math.radians(90) - (max_elevation_rad + math.radians(j))) + math.sin(math.radians(90) - (max_elevation_rad + math.radians(j))) / math.tan(max_elevation_rad + math.radians(j))
        for i in range(len(Az)):
            Cr = 100 / D * (100 - 100 * Az[i]) / 10000
            coverage_ratios_and_D_values[(j, Az[i])] = (Cr, D)
    
    return coverage_ratios_and_D_values

# Calculate the coverage ratios and D values
coverage_ratios_and_D_values = calculate_coverage_ratio(reference_angle_EL)

# Convert the dictionary to a DataFrame
df = pd.DataFrame(list(coverage_ratios_and_D_values.items()), columns=['j', 'Cr_D'])
df[['Cr', 'D']] = pd.DataFrame(df['Cr_D'].tolist(), index=df.index)
df = df.drop(columns=['Cr_D'])

# Save the DataFrame to a CSV file
df.to_csv('......./Production Model/coverage_ratios.csv', index=False)

# Print the reference_angle_EL
print(f"Reference Angle EL: {reference_angle_EL}")

# ---------------------------------------------------------------------
# 9- calculate power output for each combination of tolerance using one of the model approaches.
# ---------------------------------------------------------------------

# CALCULATE POWER OUTPUT : MODEL 1

# Load necessary files
def calculate_output_power(dni, dhi, result, cos_azimuth):
    # Constants

    # Model 0 (commented out)
    #dni_factor = 0.65 * 6 + 0.8
    #dhi_factor = 0.65 * 6 * 0.3 + 0.8
    #power_conversion_factor = 30 / 1000
    #efficiency_factor = 0.95 / 3

    # Model 1 (commented out)
    #dni_factor = 0.077
    #dhi_factor = 0.163
    #power_conversion_factor = 1
    #efficiency_factor = 0.95 / 3

    # Model 2
    dni_factor = 0.127
    dhi_factor = 0.115 
    power_conversion_factor = 1
    efficiency_factor = 0.95 / 3

    # Model 3 (commented out)
    #dni_factor = 0.127
    #dhi_factor = 0.163 
    #power_conversion_factor = 1
    #efficiency_factor = 0.95 / 3

    # Model 4 (commented out)
    #dni_factor = 0.077
    #dhi_factor = 0.114 
    #power_conversion_factor = 1
    #efficiency_factor = 0.95 / 3

    # Calculate DNI contribution
    dni_contribution = dni * dni_factor * cos_azimuth * result

    # Calculate DHI contribution
    dhi_contribution = dhi * dhi_factor

    # Combine contributions and apply conversion and efficiency factors
    power_output = (power_conversion_factor * (dni_contribution + dhi_contribution)) * efficiency_factor

    return power_output

# Load necessary files
def load_matrix(file_path):
    try:
        return pd.read_csv(file_path, header=0, index_col=0).apply(pd.to_numeric, errors='coerce').values
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

dhi_matrix = load_matrix('......./Production Model/DHI_Matrix_20min.csv')
dni_matrix = load_matrix('......./DNI_Matrix_20min.csv')
cos_azimuth_matrix = load_matrix('......./Cos_Azimuth_Angles.csv')

# Load coverage ratios and inspect the DataFrame, skipping the first row
coverage_ratio_df = pd.read_csv('......./coverage_ratios.csv')
print("Coverage Ratio DataFrame:")
print(coverage_ratio_df.head())  # Print the first few rows to inspect

# Correct the key error if necessary
coverage_ratios = dict(zip(coverage_ratio_df['j'], coverage_ratio_df['Cr']))

if dhi_matrix is None or dni_matrix is None or cos_azimuth_matrix is None:
    print("Error loading one or more matrices. Exiting.")
    exit(1)

elevation_values = range(11)

# ---------------------------------------------------------------------
# 10- loop through each azimuth and elevation combination, compute power output, and save to CSV.
# ---------------------------------------------------------------------

for azimuth in tolerance_values:
    azimuth_index = tolerance_values.index(azimuth)
    for elevation in elevation_values:
        file_path = f'......./Result_Azimuth_{azimuth}_Elevation_{elevation}.csv'
        result_matrix = load_matrix(file_path)
        
        if result_matrix is None:
            print(f"Error loading result matrix for azimuth {azimuth} and elevation {elevation}.")
            continue
        
        try:
            power_output_matrix = calculate_output_power(dni_matrix, dhi_matrix, result_matrix, cos_azimuth_matrix)
            output_file_path = f'......./Power_Output_Azimuth_{azimuth}_Elevation_{elevation}.csv'
            pd.DataFrame(power_output_matrix).to_csv(output_file_path, header=None, index=None)
        except Exception as e:
            print(f"An error occurred while calculating power output for azimuth {azimuth} and elevation {elevation}: {e}")


# ---------------------------------------------------------------------
# 11- combine the power outputs with coverage ratios to get total power, kWp, and kWh/kWp.
# ---------------------------------------------------------------------
            
def calculate_total_power(tolerance_values, coverage_ratios_file):
    total_power_outputs = []
    # Initialize total power output
    total_power_output = 0  

    # Create a dictionary that maps the azimuth tolerances in tolerance_values to the ones in the CSV file
    tolerance_mapping = {tolerance: str(i * 0.01) for i, tolerance in enumerate(tolerance_values)}

    # Load the coverage ratios from the CSV file
    coverage_ratios_df = pd.read_csv(coverage_ratios_file)
    coverage_ratios = coverage_ratios_df.set_index('j').to_dict()['Cr']

    # Iterate over all tolerance values
    for tolerance_EL in range(11):
        for tolerance in tolerance_values:
            # Get the corresponding azimuth tolerance in the CSV file
            csv_tolerance = tolerance_mapping[tolerance]

            # Calculate the coverage ratio for the current tolerance
            key = f"({tolerance_EL}, {csv_tolerance})"
            coverage_ratio = coverage_ratios.get(key, 0)  # Use a default value of 0 if the key is not found

            # Load the power output matrix from the CSV file
            file_path = f'......./Power_Output_Azimuth_{tolerance}_Elevation_{tolerance_EL}.csv'
            power_output_matrix = pd.read_csv(file_path, header=None).values

            # Sum all the values in the matrix
            summed_power_output = np.sum(power_output_matrix) * 30.4

            # Multiply the sum by the coverage ratio and add it to the total power output
            total_power_output = summed_power_output * coverage_ratio * 10000 / 1000000

            # Calculate kWp and kWh/kWp
            kWp = 1000 * 0.2 * (0.65 + 0.8 * 0.12) * 10000 * coverage_ratio / 1000
            #kWp = 140 * 10000 * coverage_ratio / 1000
            kWh_per_kWp = total_power_output * 1000 / kWp if kWp != 0 else 0

            # Store the total power output, the coverage ratio, kWp and kWh/kWp in the list
            total_power_outputs.append([f'Azimuth_{tolerance}_Elevation_{tolerance_EL}', total_power_output, coverage_ratio, kWp, kWh_per_kWp])

    return total_power_outputs


coverage_ratios_file = '......./coverage_ratios.csv'

total_power_outputs = calculate_total_power(tolerance_values, coverage_ratios_file)

# Convert the list to a DataFrame and save it to a CSV file
total_power_outputs_df = pd.DataFrame(total_power_outputs, columns=['Tolerance', 'Total Power Output', 'Coverage Ratio', 'kWp', 'kWh/kWp'])
total_power_outputs_df.to_csv('......./total_power_outputs.csv', index=False)

# Find the row with the maximum power output
max_power_output_row = total_power_outputs_df.loc[total_power_outputs_df['Total Power Output'].idxmax()]

# Print the maximum power output, its corresponding tolerance combination, kWp, kWh/kWp and Coverage Ratio
print(f"Maximum Power Output: {max_power_output_row['Total Power Output']} MW")
print(f"Tolerance Combination: {max_power_output_row['Tolerance']}")
print(f"kWp: {max_power_output_row['kWp']}")
print(f"kWh/kWp: {max_power_output_row['kWh/kWp']}")
print(f"Coverage Ratio: {max_power_output_row['Coverage Ratio']}")
