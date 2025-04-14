import pandas as pd
import numpy as np
from functools import lru_cache
import os
import time
import pulp  # For Integer Programming


import streamlit as st

st.set_page_config(page_title="Baggage Optimization", layout="wide")
st.title("Travel Baggage Optimization")
st.write("This application helps you optimize which items to pack in your cabin baggage, check-in baggage, or send via packers and movers.")

class PackingOptimizer:
    def __init__(self, excel_file_path, cabin_weight_limit, cabin_volume_limit,
                 checkin_weight_limit, checkin_volume_limit, packer_cost_per_liter,
                 value_weight_ratio=0.5, value_volume_ratio=0.5):
        """
        Initialize the packing optimizer with constraints and item data from Excel file.

        Parameters:
        -----------
        excel_file_path : str
            Path to the Excel file containing item details
        cabin_weight_limit : float
            Maximum weight allowed in cabin baggage (kg)
        cabin_volume_limit : float
            Maximum volume allowed in cabin baggage (liters)
        checkin_weight_limit : float
            Maximum weight allowed in check-in baggage (kg)
        checkin_volume_limit : float
            Maximum volume allowed in check-in baggage (liters)
        packer_cost_per_liter : float
            Cost per liter for packers and movers (INR)
        value_weight_ratio : float
            Importance of value-per-weight in optimization (0.0 to 1.0)
        value_volume_ratio : float
            Importance of value-per-volume in optimization (0.0 to 1.0)
        """
        # Load data from Excel file
        self.items_data = pd.read_excel(excel_file_path)

        # Process the data into our format
        self.items = self._preprocess_items(self.items_data)

        # Calculate value-per-weight and value-per-volume for each item
        for item in self.items:
            item['value_per_weight'] = item['value'] / max(0.1, item['weight'])  # Avoid division by zero
            item['value_per_volume'] = item['value'] / max(0.1, item['volume'])  # Avoid division by zero

            # Calculate combined efficiency score
            item['efficiency_score'] = (
                value_weight_ratio * item['value_per_weight'] +
                value_volume_ratio * item['value_per_volume']
            )

        # Store importance ratios
        self.value_weight_ratio = value_weight_ratio
        self.value_volume_ratio = value_volume_ratio

        # Store constraints
        self.cabin_weight_limit = cabin_weight_limit
        self.cabin_volume_limit = cabin_volume_limit
        self.checkin_weight_limit = checkin_weight_limit
        self.checkin_volume_limit = checkin_volume_limit
        self.packer_cost_per_liter = packer_cost_per_liter

        # We'll use discretization for weight and volume to make the state space manageable
        # Each unit represents a fixed amount of weight/volume
        self.weight_unit = 0.5  # 0.5 kg per unit
        self.volume_unit = 1.0  # 1.0 liter per unit

        # Convert limits to discrete units
        self.cabin_weight_units = int(self.cabin_weight_limit / self.weight_unit)
        self.cabin_volume_units = int(self.cabin_volume_limit / self.volume_unit)
        self.checkin_weight_units = int(self.checkin_weight_limit / self.weight_unit)
        self.checkin_volume_units = int(self.checkin_volume_limit / self.volume_unit)

        # Total number of items
        self.n_items = len(self.items)

        # Memoization dictionary to store computed results
        self.memo = {}

        # To track the placement of each item for the optimal solution
        self.placement = {}

        # For tracking solution details in DP
        self.best_placement_cache = {}

    def _preprocess_items(self, items_data):
        """
        Preprocess items data into a list of dictionaries with required fields.
        """
        items = []

        # Determine column names based on the Excel file structure
        # This handles both "Item" and "ItemName" column naming conventions
        name_col = next((col for col in items_data.columns if 'item' in col.lower()), None)

        # If no explicit name column, try to use the first column
        if name_col is None and not items_data.empty:
            name_col = items_data.columns[0]

        # Find value/price/cost column
        value_col = next((col for col in items_data.columns if 'value' in col.lower() or 'price' in col.lower() or 'cost' in col.lower() or 'monetary' in col.lower()), None)
        
        # Define column mappings
        col_mapping = {
            'name': name_col,
            'weight': next((col for col in items_data.columns if 'weight' in col.lower()), 'Weight (kg)'),
            'volume': next((col for col in items_data.columns if 'volume' in col.lower() or 'vol' in col.lower()), 'Volume (liters)'),
            'value': value_col or 'Current Monetary Value (INR)',  # Use detected column or fallback
            'baggage_type': next((col for col in items_data.columns if 'baggage' in col.lower() or 'type' in col.lower()), 'Airline Baggage Type')
        }
        
        # Fallback values if columns aren't found
        default_values = {
            'weight': 1.0,    # Default 1kg
            'volume': 2.0,    # Default 2 liters
            'value': 1000.0,  # Default 1000 INR
        }

        for idx, row in items_data.iterrows():
            # Extract item name
            if col_mapping['name'] and col_mapping['name'] in items_data.columns:
                item_name = str(row[col_mapping['name']])
            else:
                item_name = f'Item_{idx}'

            # Create item dictionary with values from appropriate columns, or defaults
            item = {
                'name': item_name,
                'id': idx
            }
            
            # Extract weight, volume, value with fallbacks
            for field in ['weight', 'volume', 'value']:
                if col_mapping[field] and col_mapping[field] in items_data.columns:
                    try:
                        # Try to convert to float, use default if fails
                        item[field] = float(row[col_mapping[field]])
                    except (ValueError, TypeError):
                        item[field] = default_values[field]
                else:
                    item[field] = default_values[field]
            
            # Extract baggage type or use default
            if col_mapping['baggage_type'] and col_mapping['baggage_type'] in items_data.columns:
                item['baggage_type'] = str(row[col_mapping['baggage_type']])
            else:
                item['baggage_type'] = 'Both'  # Default to both types

            # Set compatibility flags - fixed to handle "Both" baggage type
            item['cabin_compatible'] = item['baggage_type'] in ['Hand Baggage', 'Both']
            item['checkin_compatible'] = item['baggage_type'] in ['Check-in Baggage', 'Both']
            item['packer_compatible'] = True  # Assume all items can be sent via packers by default

            items.append(item)
            
        return items

    def optimize_with_dp(self):
        """
        Run the dynamic programming optimization that balances packer costs,
        value-per-weight, and value-per-volume efficiency.
        """
        # Convert item weights and volumes to discrete units
        for item in self.items:
            item['weight_units'] = max(1, int(np.ceil(item['weight'] / self.weight_unit)))
            item['volume_units'] = max(1, int(np.ceil(item['volume'] / self.volume_unit)))

        # Clear any previous placements and memoization
        self.placement = {}
        self.memo = {}
        self.best_placement_cache = {}

        # Start the recursion from the first item with empty baggage
        optimal_value = self._optimal_value_dp(0, 0, 0, 0, 0)

        # Reconstruct the solution using our placement cache
        self._reconstruct_solution(0, 0, 0, 0, 0)

        # Extract items for each baggage type
        cabin_items = []
        checkin_items = []
        packer_items = []

        # Create item lookup by ID
        item_lookup = {item['id']: item for item in self.items}

        for item_id, placement in self.placement.items():
            item = item_lookup.get(item_id)
            if item:
                if placement == 'cabin':
                    cabin_items.append(item)
                elif placement == 'checkin':
                    checkin_items.append(item)
                elif placement == 'packer':
                    packer_items.append(item)

        # Calculate baggage weights, volumes, and values
        cabin_weight = sum(item['weight'] for item in cabin_items)
        cabin_volume = sum(item['volume'] for item in cabin_items)
        checkin_weight = sum(item['weight'] for item in checkin_items)
        checkin_volume = sum(item['volume'] for item in checkin_items)
        packer_volume = sum(item['volume'] for item in packer_items)
        packer_cost = packer_volume * self.packer_cost_per_liter

        # Calculate total value for each section
        cabin_value = sum(item['value'] for item in cabin_items)
        cabin_bonus = sum(500 * item['efficiency_score'] for item in cabin_items)
        
        checkin_value = sum(item['value'] for item in checkin_items)
        checkin_bonus = sum(300 * item['efficiency_score'] for item in checkin_items)
        
        packer_value = sum(item['value'] for item in packer_items)
        packer_penalty = packer_cost
        
        # Calculate objective value including bonuses and penalties
        effective_value = (cabin_value + cabin_bonus) + (checkin_value + checkin_bonus) + (packer_value - packer_penalty)
        
        # For consistency with UI display, we'll keep total_value as just the sum of item values
        total_value = cabin_value + checkin_value + packer_value

        # Calculate efficiency metrics
        cabin_value_per_weight = cabin_value / max(0.1, cabin_weight) if cabin_items else 0
        cabin_value_per_volume = cabin_value / max(0.1, cabin_volume) if cabin_items else 0
        checkin_value_per_weight = checkin_value / max(0.1, checkin_weight) if checkin_items else 0
        checkin_value_per_volume = checkin_value / max(0.1, checkin_volume) if checkin_items else 0

        return {
            'total_cost': packer_cost,
            'total_value': total_value,
            'effective_value': effective_value,  # This includes the bonuses and penalties used in optimization
            'cabin_items': cabin_items,
            'checkin_items': checkin_items,
            'packer_items': packer_items,
            'cabin_weight': cabin_weight,
            'cabin_volume': cabin_volume,
            'cabin_value': cabin_value,
            'cabin_value_per_weight': cabin_value_per_weight,
            'cabin_value_per_volume': cabin_value_per_volume,
            'checkin_weight': checkin_weight,
            'checkin_volume': checkin_volume,
            'checkin_value': checkin_value,
            'checkin_value_per_weight': checkin_value_per_weight,
            'checkin_value_per_volume': checkin_value_per_volume,
            'packer_volume': packer_volume,
            'packer_value': packer_value,
            'packer_cost': packer_cost
        }

    def _optimal_value_dp(self, item_idx, cabin_w, cabin_v, checkin_w, checkin_v):
        """
        Recursive DP function with memoization to find optimal packing plan.
        This uses a combined objective that balances minimizing packer costs and
        maximizing value-per-weight and value-per-volume efficiency.
        """
        # Base case: all items considered
        if item_idx >= self.n_items:
            return 0

        # Check if we've already computed this state
        state = (item_idx, cabin_w, cabin_v, checkin_w, checkin_v)
        if state in self.memo:
            return self.memo[state]

        current_item = self.items[item_idx]
        item_weight_units = current_item['weight_units']
        item_volume_units = current_item['volume_units']

        # Initialize with a large negative value (as we're maximizing)
        best_value = float('-inf')
        best_placement = None

        # Define a "benefit" function for each placement option
        # Higher is better, combining item value and efficiency

        # Option 1: Try cabin if compatible and within limits
        if (current_item['cabin_compatible'] and
                cabin_w + item_weight_units <= self.cabin_weight_units and
                cabin_v + item_volume_units <= self.cabin_volume_units):

            # Consider both item value and efficiency score for cabin placement
            cabin_benefit = current_item['value'] + 500 * current_item['efficiency_score']

            future_value = self._optimal_value_dp(
                item_idx + 1,
                cabin_w + item_weight_units,
                cabin_v + item_volume_units,
                checkin_w,
                checkin_v
            )

            option1_value = cabin_benefit + future_value

            if option1_value > best_value:
                best_value = option1_value
                best_placement = 'cabin'

        # Option 2: Try check-in if compatible and within limits
        if (current_item['checkin_compatible'] and
                checkin_w + item_weight_units <= self.checkin_weight_units and
                checkin_v + item_volume_units <= self.checkin_volume_units):

            # Consider both item value and efficiency score for checkin placement
            checkin_benefit = current_item['value'] + 300 * current_item['efficiency_score']

            future_value = self._optimal_value_dp(
                item_idx + 1,
                cabin_w,
                cabin_v,
                checkin_w + item_weight_units,
                checkin_v + item_volume_units
            )

            option2_value = checkin_benefit + future_value

            if option2_value > best_value:
                best_value = option2_value
                best_placement = 'checkin'

        # Option 3: Send via packers
        if current_item['packer_compatible']:
            # Apply a penalty for using packers based on cost
            packer_penalty = current_item['volume'] * self.packer_cost_per_liter

            future_value = self._optimal_value_dp(
                item_idx + 1,
                cabin_w,
                cabin_v,
                checkin_w,
                checkin_v
            )

            # We still get the item value but with a cost penalty
            option3_value = current_item['value'] - packer_penalty + future_value

            if option3_value > best_value:
                best_value = option3_value
                best_placement = 'packer'

        # Store the best placement for this state
        self.best_placement_cache[state] = best_placement

        # Memoize and return
        self.memo[state] = best_value
        return best_value

    def _reconstruct_solution(self, item_idx, cabin_w, cabin_v, checkin_w, checkin_v):
        """
        Reconstruct the solution by following the choices made during DP.
        """
        if item_idx >= self.n_items:
            return

        state = (item_idx, cabin_w, cabin_v, checkin_w, checkin_v)
        if state not in self.best_placement_cache:
            return

        placement = self.best_placement_cache[state]
        current_item = self.items[item_idx]

        self.placement[current_item['id']] = placement

        if placement == 'cabin':
            self._reconstruct_solution(
                item_idx + 1,
                cabin_w + current_item['weight_units'],
                cabin_v + current_item['volume_units'],
                checkin_w,
                checkin_v
            )
        elif placement == 'checkin':
            self._reconstruct_solution(
                item_idx + 1,
                cabin_w,
                cabin_v,
                checkin_w + current_item['weight_units'],
                checkin_v + current_item['volume_units']
            )
        else:  # packer
            self._reconstruct_solution(
                item_idx + 1,
                cabin_w,
                cabin_v,
                checkin_w,
                checkin_v
            )

    def optimize_with_ip(self):
        """
        Run the optimization using Integer Programming (IP) approach.
        This method balances minimizing packer costs and maximizing value efficiency.
        """
        # Create a new LP problem
        model = pulp.LpProblem(name="Baggage_Packing", sense=pulp.LpMaximize)

        # Create binary decision variables for each item and baggage type
        x_cabin = {i: pulp.LpVariable(f"x_cabin_{i}", cat=pulp.LpBinary)
                  for i in range(self.n_items)}

        x_checkin = {i: pulp.LpVariable(f"x_checkin_{i}", cat=pulp.LpBinary)
                    for i in range(self.n_items)}

        x_packer = {i: pulp.LpVariable(f"x_packer_{i}", cat=pulp.LpBinary)
                   for i in range(self.n_items)}

        # Define the objective function:
        # 1. Maximize total value
        # 2. Consider value-per-weight and value-per-volume efficiency
        # 3. Minimize packer costs

        # Efficiency bonuses for cabin and check-in items
        cabin_efficiency_bonus = pulp.lpSum([
            (500 * self.items[i]['efficiency_score'] * x_cabin[i])
            for i in range(self.n_items)
        ])

        checkin_efficiency_bonus = pulp.lpSum([
            (300 * self.items[i]['efficiency_score'] * x_checkin[i])
            for i in range(self.n_items)
        ])

        # Packer cost penalty
        packer_cost_penalty = pulp.lpSum([
            self.items[i]['volume'] * self.packer_cost_per_liter * x_packer[i]
            for i in range(self.n_items)
        ])

        # Total item value
        total_value = pulp.lpSum([
            self.items[i]['value'] * (x_cabin[i] + x_checkin[i] + x_packer[i])
            for i in range(self.n_items)
        ])

        # Combined objective function
        model += total_value + cabin_efficiency_bonus + checkin_efficiency_bonus - packer_cost_penalty

        # Constraint 1: Each item must be placed somewhere
        for i in range(self.n_items):
            model += x_cabin[i] + x_checkin[i] + x_packer[i] == 1

        # Constraint 2: Cabin weight limit
        model += pulp.lpSum([self.items[i]['weight'] * x_cabin[i]
                             for i in range(self.n_items)]) <= self.cabin_weight_limit

        # Constraint 3: Cabin volume limit
        model += pulp.lpSum([self.items[i]['volume'] * x_cabin[i]
                             for i in range(self.n_items)]) <= self.cabin_volume_limit

        # Constraint 4: Check-in weight limit
        model += pulp.lpSum([self.items[i]['weight'] * x_checkin[i]
                             for i in range(self.n_items)]) <= self.checkin_weight_limit

        # Constraint 5: Check-in volume limit
        model += pulp.lpSum([self.items[i]['volume'] * x_checkin[i]
                             for i in range(self.n_items)]) <= self.checkin_volume_limit

        # Constraint 6: Cabin compatibility
        for i in range(self.n_items):
            if not self.items[i]['cabin_compatible']:
                model += x_cabin[i] == 0

        # Constraint 7: Check-in compatibility
        for i in range(self.n_items):
            if not self.items[i]['checkin_compatible']:
                model += x_checkin[i] == 0

        # Constraint 8: Packer compatibility
        for i in range(self.n_items):
            if not self.items[i]['packer_compatible']:
                model += x_packer[i] == 0

        # Solve the model
        model.solve(pulp.PULP_CBC_CMD(msg=False))

        # Extract the solution
        cabin_items = []
        checkin_items = []
        packer_items = []

        for i in range(self.n_items):
            if pulp.value(x_cabin[i]) == 1:
                cabin_items.append(self.items[i])
                self.placement[self.items[i]['id']] = 'cabin'
            elif pulp.value(x_checkin[i]) == 1:
                checkin_items.append(self.items[i])
                self.placement[self.items[i]['id']] = 'checkin'
            elif pulp.value(x_packer[i]) == 1:
                packer_items.append(self.items[i])
                self.placement[self.items[i]['id']] = 'packer'

        # Calculate total weights, volumes, values, and costs
        cabin_weight = sum(item['weight'] for item in cabin_items)
        cabin_volume = sum(item['volume'] for item in cabin_items)
        checkin_weight = sum(item['weight'] for item in checkin_items)
        checkin_volume = sum(item['volume'] for item in checkin_items)
        packer_volume = sum(item['volume'] for item in packer_items)
        packer_cost = packer_volume * self.packer_cost_per_liter

        # Calculate total value for each section
        cabin_value = sum(item['value'] for item in cabin_items)
        cabin_bonus = sum(500 * item['efficiency_score'] for item in cabin_items)
        
        checkin_value = sum(item['value'] for item in checkin_items)
        checkin_bonus = sum(300 * item['efficiency_score'] for item in checkin_items)
        
        packer_value = sum(item['value'] for item in packer_items)
        packer_penalty = packer_cost
        
        # Calculate objective value including bonuses and penalties
        effective_value = (cabin_value + cabin_bonus) + (checkin_value + checkin_bonus) + (packer_value - packer_penalty)
        
        # For consistency with UI display, keep total_value as just the sum of item values
        total_value = cabin_value + checkin_value + packer_value

        # Calculate efficiency metrics
        cabin_value_per_weight = cabin_value / max(0.1, cabin_weight) if cabin_items else 0
        cabin_value_per_volume = cabin_value / max(0.1, cabin_volume) if cabin_items else 0
        checkin_value_per_weight = checkin_value / max(0.1, checkin_weight) if checkin_items else 0
        checkin_value_per_volume = checkin_value / max(0.1, checkin_volume) if checkin_items else 0

        return {
            'total_cost': packer_cost,
            'total_value': total_value,
            'effective_value': effective_value,  # This includes the bonuses and penalties used in optimization
            'cabin_items': cabin_items,
            'checkin_items': checkin_items,
            'packer_items': packer_items,
            'cabin_weight': cabin_weight,
            'cabin_volume': cabin_volume,
            'cabin_value': cabin_value,
            'cabin_value_per_weight': cabin_value_per_weight,
            'cabin_value_per_volume': cabin_value_per_volume,
            'checkin_weight': checkin_weight,
            'checkin_volume': checkin_volume,
            'checkin_value': checkin_value,
            'checkin_value_per_weight': checkin_value_per_weight,
            'checkin_value_per_volume': checkin_value_per_volume,
            'packer_volume': packer_volume,
            'packer_value': packer_value,
            'packer_cost': packer_cost
        }


# Streamlit app implementation
def main():
    st.sidebar.header("Baggage Constraints")
    
    # Input parameters in the sidebar
    cabin_weight_limit = st.sidebar.number_input("Cabin Weight Limit (kg)", min_value=1.0, max_value=20.0, value=7.0, step=0.5)
    cabin_volume_limit = st.sidebar.number_input("Cabin Volume Limit (liters)", min_value=10.0, max_value=100.0, value=44.0, step=1.0)
    checkin_weight_limit = st.sidebar.number_input("Check-in Weight Limit (kg)", min_value=5.0, max_value=50.0, value=25.0, step=1.0)
    checkin_volume_limit = st.sidebar.number_input("Check-in Volume Limit (liters)", min_value=50.0, max_value=200.0, value=116.13, step=1.0)
    packer_cost = st.sidebar.number_input("Packer Cost per Liter (INR)", min_value=1.0, max_value=500.0, value=50.0, step=5.0)
    
    # Value efficiency weighting parameters
    st.sidebar.header("Optimization Parameters")
    value_weight_ratio = st.sidebar.slider("Value-per-Weight Importance", min_value=0.0, max_value=1.0, value=0.6, step=0.1)
    value_volume_ratio = st.sidebar.slider("Value-per-Volume Importance", min_value=0.0, max_value=1.0, value=0.4, step=0.1)
    
    # File uploader
    st.header("Upload Your Items Data")
    uploaded_file = st.file_uploader("Choose an Excel file with your items", type=["xlsx", "xls"])
    
    if uploaded_file is not None:
        # Display a preview of the uploaded data
        df = pd.read_excel(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())
        
        # Display the dimensions of the dataframe
        st.write(f"Total items: {len(df)}")
        
        # Debug column information
        st.subheader("Debug Information")
        st.write("Column names in the uploaded file:")
        st.write(df.columns.tolist())
        
        # Check for value/monetary columns
        value_cols = [col for col in df.columns if 'value' in col.lower() or 'price' in col.lower() or 'cost' in col.lower() or 'monetary' in col.lower()]
        if value_cols:
            st.write("Possible value columns found:", value_cols)
            
            # Show sample of first value column
            if value_cols[0] in df.columns:
                st.write(f"Sample values from '{value_cols[0]}':")
                st.write(df[value_cols[0]].head())
                st.write(f"Data type: {df[value_cols[0]].dtype}")
        else:
            st.warning("No value/price/cost columns detected in the data. Please ensure your file has a column for item values.")
        
        # Optimization method selection
        optimization_method = st.radio(
            "Select Optimization Method",
            ["Integer Programming (IP)", "Dynamic Programming (DP)"]
        )
        
        if st.button("Run Optimization"):
            with st.spinner("Optimizing your baggage allocation..."):
                # Create optimizer with efficiency parameters
                optimizer = PackingOptimizer(
                    uploaded_file,
                    cabin_weight_limit,
                    cabin_volume_limit,
                    checkin_weight_limit,
                    checkin_volume_limit,
                    packer_cost,
                    value_weight_ratio,
                    value_volume_ratio
                )
                
                # Start timer
                start_time = time.time()
                
                # Run optimization based on selected method
                if optimization_method == "Integer Programming (IP)":
                    result = optimizer.optimize_with_ip()
                    method_name = "IP-Based Optimization"
                else:
                    result = optimizer.optimize_with_dp()
                    method_name = "DP-Based Optimization"
                
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Display results
                st.header(f"Results ({method_name})")
                st.write(f"Execution time: {execution_time:.4f} seconds")
                
                # Summary metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Summary")
                    st.write(f"Total value of all items: ₹{result['total_value']:.2f}")
                    st.write(f"Total packer cost: ₹{result['packer_cost']:.2f}")
                    st.write(f"Net value (after packer costs): ₹{result['total_value'] - result['packer_cost']:.2f}")
                
                with col2:
                    st.subheader("Baggage Utilization")
                    st.write(f"Cabin: {result['cabin_weight']:.2f}/{cabin_weight_limit} kg, {result['cabin_volume']:.2f}/{cabin_volume_limit} L")
                    st.write(f"Check-in: {result['checkin_weight']:.2f}/{checkin_weight_limit} kg, {result['checkin_volume']:.2f}/{checkin_volume_limit} L")
                    st.write(f"Packers: {result['packer_volume']:.2f} L at ₹{packer_cost}/L = ₹{result['packer_cost']:.2f}")
                
                # Display the optimization objective value
                st.info(f"Optimization Score: ₹{result['effective_value']:.2f} (includes efficiency bonuses and penalties used by the algorithm)")
                
                # Tabs for detailed results
                tab1, tab2, tab3 = st.tabs(["Cabin Items", "Check-in Items", "Packer Items"])
                
                with tab1:
                    st.subheader("Cabin Baggage Items")
                    st.write(f"Total items: {len(result['cabin_items'])}")
                    st.write(f"Total weight: {result['cabin_weight']:.2f} kg")
                    st.write(f"Total volume: {result['cabin_volume']:.2f} liters")
                    st.write(f"Total value: ₹{result['cabin_value']:.2f}")
                    
                    if result['cabin_items']:
                        cabin_df = pd.DataFrame(result['cabin_items'])
                        st.dataframe(cabin_df[['name', 'weight', 'volume', 'value']])
                    else:
                        st.write("No items allocated to cabin baggage.")
                
                with tab2:
                    st.subheader("Check-in Baggage Items")
                    st.write(f"Total items: {len(result['checkin_items'])}")
                    st.write(f"Total weight: {result['checkin_weight']:.2f} kg")
                    st.write(f"Total volume: {result['checkin_volume']:.2f} liters")
                    st.write(f"Total value: ₹{result['checkin_value']:.2f}")
                    
                    if result['checkin_items']:
                        checkin_df = pd.DataFrame(result['checkin_items'])
                        st.dataframe(checkin_df[['name', 'weight', 'volume', 'value']])
                    else:
                        st.write("No items allocated to check-in baggage.")
                
                with tab3:
                    st.subheader("Items via Packers & Movers")
                    st.write(f"Total items: {len(result['packer_items'])}")
                    st.write(f"Total volume: {result['packer_volume']:.2f} liters")
                    st.write(f"Total value: ₹{result['packer_value']:.2f}")
                    st.write(f"Total cost: ₹{result['packer_cost']:.2f}")
                    
                    if result['packer_items']:
                        packer_df = pd.DataFrame(result['packer_items'])
                        packer_df['cost'] = packer_df['volume'] * packer_cost
                        st.dataframe(packer_df[['name', 'weight', 'volume', 'value', 'cost']])
                    else:
                        st.write("No items allocated to packers & movers.")
    else:
        st.info("Please upload an Excel file with your items to start the optimization.")
        st.write("The Excel file should contain columns for item name, weight (kg), volume (liters), value (INR), and baggage type.")
        
        # Sample data format
        st.subheader("Expected data format:")
        sample_data = {
            'Item': ['Laptop', 'Clothes', 'Books', 'Furniture'],
            'Weight (kg)': [2.5, 8.0, 5.0, 35.0],
            'Volume (liters)': [3.0, 15.0, 8.0, 75.0],
            'Current Monetary Value (INR)': [80000, 12000, 5000, 45000],
            'Airline Baggage Type': ['Hand Baggage', 'Check-in Baggage', 'Both', 'Check-in Baggage']
        }
        
        st.dataframe(pd.DataFrame(sample_data))

if __name__ == "__main__":
    main()