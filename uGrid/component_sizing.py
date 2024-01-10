import pandas as pd
import numpy as np
from math import ceil
from openpyxl import Workbook
from datetime import datetime

# Minigrid Sizing Script
# Description:
# This script performs sizing calculations for a solar minigrid system based on site-specific requirements.
# It reads input data from an Excel file, which includes details for PV panels, inverters, batteries, 
# and site-specific energy needs. The script calculates the number and configuration of components required 
# to meet these energy needs. The output is an Excel file detailing the system configuration.
# The input file has separate tabs for different data types: site data, PV specifications, inverter specifications, 
# and battery specifications.

# Hardcoded data
def get_hardcoded_data():
    inverter_data = pd.DataFrame({
        'Inverter Model': ['SUN-25K-SG01HP3-EU-BM2', 'SUN-30K-SG01HP3-EU-BM3', 'SUN-40K-SG01HP3-EU-BM4', 'SUN-50K-SG01HP3-EU-BM4', 'SUN-20K-SG01HP3-EU-AM2', 'SUN-35K-SG01HP3-EU-BM3'],
        'Inverter Vmax (V)': [850, 850, 850, 850, 850, 850],
        'Inverter Imax (A)': [55, 55, 55, 55, 39, 55],
        'Inverter #MPPT inputs': [2, 3, 4, 4, 2, 3],
        'Strings Per MPPT': [2, 2, 2, 2, 2, 2],
        'Inverter kWac': [25, 30, 40, 50, 20, 35]
    })
    battery_data = pd.DataFrame({
        'Battery Model': ['HV4875', 'HV51100'],
        'V Batt': [48, 51.2],
        'kWh': [3.6, 5.12],
        'Rack Sizes': [[7, 11], [7, 11]]  # Add rack sizes for each battery model
    })
    panel_data = pd.DataFrame({
        'Panel Model': ['XYZ-400W'],
        'Panel Voc (V)': [38],
        'Panel Isc (A)': [19],
        'Panel Wp (W)': [650]
    })
    site_data = pd.DataFrame({
        'Site Name': ['Site A', 'Site B'],
        'Target PVkW': [50, 60],
        'Target kWh': [100, 150]
    })
    return inverter_data, battery_data, panel_data, site_data

# Load data from Excel file or use hardcoded data
def load_data(input_file):
    try:
        data = pd.read_excel(input_file, sheet_name=None)
        inverter_data = data['Inverters']
        battery_data = data['Batteries']
        panel_data = data['PV Panels']
        site_data = data['Sites']
    except Exception as e:
        print(f"Error loading input file: {e}. Using hardcoded data.")
        inverter_data, battery_data, panel_data, site_data = get_hardcoded_data()
    return inverter_data, battery_data, panel_data, site_data

# Calculate panel configuration
def calculate_panel_config(target_pvkW, panel_Wp, chosen_inverter, panel_Voc):
    num_panels = round(target_pvkW * 1000 / panel_Wp)

    max_panels_per_string = int(chosen_inverter['Inverter Vmax (V)'] / panel_Voc)
    num_type_A_strings = num_panels // max_panels_per_string
    num_type_B_strings = 0
    panels_in_type_B_string = 0

    remaining_panels = num_panels - (num_type_A_strings * max_panels_per_string)
    if remaining_panels > 0:
        num_type_B_strings = 1
        panels_in_type_B_string = remaining_panels

    return num_panels, max_panels_per_string, num_type_A_strings, num_type_B_strings, panels_in_type_B_string

# Calculate fuse and circuit breaker ratings
def calculate_ratings(panel_Isc, num_type_A_strings, num_type_B_strings):
    fuse_rating = 15 if panel_Isc <= 10 else ceil(panel_Isc / 5) * 5
    if num_type_A_strings <= 1 and num_type_B_strings <= 1:
        # If there are no strings connected in parallel
        circuit_breaker_rating = 15
    else:
        # If there are strings connected in parallel
        circuit_breaker_rating = ceil(panel_Isc * 2 / 5) * 5

    return fuse_rating, circuit_breaker_rating

# Calculate generator capacity and circuit breaker rating
def calculate_generator_specs(actual_ackW):
    generator_capacity = round(2 / 3 * actual_ackW, -1)
    generator_model = f"ABD-{generator_capacity}GF"
    generator_circuit_breaker_rating = ceil((generator_capacity * 1000) / (3 * 220) / 5) * 5  # convert kW to W

    return generator_capacity, generator_model, generator_circuit_breaker_rating

# Calculate battery configuration
def calculate_battery_config(target_kWh, battery_data):
    rack_sizes = [7, 11]  # Number of batteries per rack (either 7 or 11)

    best_diff = float('inf')  # Initialize with a large value
    best_config = None
    for _, battery in battery_data.iterrows():
        for rack_size in rack_sizes:
            rack_kWh = rack_size * battery['kWh']
            num_racks = target_kWh // rack_kWh
            total_kWh = num_racks * rack_kWh
            diff = target_kWh - total_kWh
            if diff < best_diff:
                best_diff = diff
                best_config = (battery['Battery Model'], rack_size, num_racks, rack_size * battery['V Batt'])

    return best_config

# Main function
def main():
    # Load data # Replace 'Component_Sizing.xlsx' with actual file path
    inverter_data, battery_data, panel_data, site_data = load_data('/Users/mattmso/Desktop/Component_Sizing_input.xlsx')  




    # Initialize a workbook and select the active sheet
    wb = Workbook()
    ws = wb.active

    # Write the headers to the sheet
    headers = ['[SITE NAME]', '[TARGET PVKW]', '[TARGET KWH]', '[PVMODEL]','[STANUM]', '[PNUMA]', '[STBNUM]', '[PNUMB]', '[PVKW]', '[ACKW]', '[FA]', '[PVDCA]', '[INVERTERNAME]', '[INVNUM]', '[INVKW]', '[CBA]', '[GKW]', '[GENNAME]', '[GCBA]', '[MAINSCB]', '[BRNUM]', '[BNUM]', '[BMODEL]', '[VB]', '[BKWH]', '[BANKKWH]', '[BDCA]']
    ws.append(headers)

    for idx, row in site_data.iterrows():
        site_name = row['Site Name']
        target_pvkW = row['Target PVkW']
        target_ackW = target_pvkW / 1.2
        target_kWh = row['Target kWh']
    
        # Retrieve panel specifications from panel_data
        Panel_Model = panel_data['Panel Model'].iloc[0]
        Panel_Wp = panel_data['Panel Wp (W)'].iloc[0]
        Panel_Isc = panel_data['Panel Isc (A)'].iloc[0]
        Panel_Voc = panel_data['Panel Voc (V)'].iloc[0]
    
        # Find the inverter model that when paralleled most closely approximates the target AC kW
        inverter_data['Paralleled Inverters'] = np.ceil(target_ackW / inverter_data['Inverter kWac'])
        inverter_data['Total AC kW'] = inverter_data['Paralleled Inverters'] * inverter_data['Inverter kWac']
        inverter_data['Difference'] = abs(inverter_data['Total AC kW'] - target_ackW)
        chosen_inverter = inverter_data.loc[inverter_data['Difference'].idxmin()]
    
        num_panels, max_panels_per_string, num_type_A_strings, num_type_B_strings, panels_in_type_B_string = calculate_panel_config(target_pvkW, Panel_Wp, chosen_inverter, Panel_Voc)
    
        num_inverters = int(chosen_inverter['Paralleled Inverters'])
    
        # Calculate the actual PV array size and the actual combined AC power output of the paralleled inverters
        actual_pvkW = num_panels * Panel_Wp / 1000
        actual_ackW = num_inverters * chosen_inverter['Inverter kWac']
    
        fuse_rating, circuit_breaker_rating = calculate_ratings(Panel_Isc, num_type_A_strings, num_type_B_strings)
    
        generator_capacity, generator_model, generator_circuit_breaker_rating = calculate_generator_specs(actual_ackW)
    
        
        # Calculate CBA
        CBA = ceil((actual_ackW * 1000) / (3 * 220) / 5) * 5  # convert kW to W

        # Calculate MAINSCB
        MAINSCB = CBA + generator_circuit_breaker_rating

        # Calculate battery configuration
        best_model, best_rack_size, best_num_racks, best_vb = calculate_battery_config(target_kWh, battery_data)

        # Calculate BKWH
        BKWH = battery_data.loc[battery_data['Battery Model'] == best_model, 'kWh'].values[0]

        # Calculate BANKKWH
        BANKKWH = best_num_racks * best_rack_size * BKWH

        # Calculate BDCA
        BDCA = ceil((actual_ackW * 1000) / (best_vb * best_num_racks) / 10) * 10  # convert kW to W

        # Write the data to the sheet
        data = [site_name, target_pvkW, target_kWh, Panel_Model, num_type_A_strings, max_panels_per_string, num_type_B_strings, panels_in_type_B_string, actual_pvkW, actual_ackW, fuse_rating, circuit_breaker_rating, chosen_inverter['Inverter Model'], num_inverters, chosen_inverter['Inverter kWac'], CBA, generator_capacity, generator_model, generator_circuit_breaker_rating, MAINSCB, best_num_racks, best_rack_size, best_model, best_vb, BKWH, BANKKWH, BDCA]
        ws.append(data)

    # Get the current date and time
    now = datetime.now()

    # Format the date and time as a string in the format YYMMDD_HHMM
    timestamp = now.strftime('%y%m%d_%H%M')

    # Append the timestamp to the filename
   
    filename = f'Component_Sizing_output_{timestamp}.xlsx'
    # Save the workbook with the new filename
    wb.save(filename)



if __name__ == "__main__":
    main()
