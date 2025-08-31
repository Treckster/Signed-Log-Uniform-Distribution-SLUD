import csv
import statistics

def calculate_statistics(csv_file, columns=None):
    """
    Read a CSV file and calculate statistics for specified columns.
    
    Args:
        csv_file (str): Path to the CSV file
        columns (list): List of column names to analyze. If None, analyzes all numeric columns.
    
    Returns:
        dict: Dictionary containing statistics for each column
    """
    # Read the CSV file and collect data
    data = {}
    headers = []
    
    with open(csv_file, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        headers = next(reader)  # Get column headers
        
        # Initialize data dictionary
        for header in headers:
            data[header] = []
        
        # Read all rows
        for row in reader:
            for i, value in enumerate(row):
                if i < len(headers):
                    # Try to convert to float, skip if not numeric
                    try:
                        numeric_value = float(value)
                        data[headers[i]].append(numeric_value)
                    except (ValueError, TypeError):
                        # Skip non-numeric values
                        continue
    
    # If no columns specified, use all columns that have numeric data
    if columns is None:
        columns = [col for col in headers if data[col]]
    
    stats = {}
    
    for col in columns:
        if col in data and data[col]:
            col_data = data[col]
            # Filter out failures (values >= 501) and count them
            failures = [x for x in col_data if x >= 501]
            success_data = [x for x in col_data if x < 501]
            failure_count = len(failures)
            
            # Use success data for max calculation, or original data if no successes
            stats[col] = {
                'count': len(success_data),  # Only successful runs
                'mean': statistics.mean(success_data) if success_data else 0,  # Only successful runs
                'median': statistics.median(success_data) if success_data else 0,  # Only successful runs
                'std': statistics.stdev(success_data) if len(success_data) > 1 else 0,  # Only successful runs
                'min': min(success_data) if success_data else 0,  # Only successful runs
                'max': max(success_data) if success_data else 0,  # Only successful runs
                'q25': statistics.quantiles(success_data, n=4)[0] if len(success_data) >= 4 else (min(success_data) if success_data else 0),  # Only successful runs
                'q75': statistics.quantiles(success_data, n=4)[2] if len(success_data) >= 4 else (max(success_data) if success_data else 0),  # Only successful runs
                'failures': failure_count,  # Count of failed runs (>= 501)
                'success percentage': (len(success_data) / len(col_data)) * 100 if col_data else 0  # Uses both successful and total runs
            }
    
    return stats

def print_statistics(stats):
    """Print statistics in a formatted way."""
    for column, values in stats.items():
        print(f"\n=== {column} ===")
        for stat_name, stat_value in values.items():
            print(f"{stat_name}: {stat_value:.4f}")

# Example usage
if __name__ == "__main__":
    # Replace 'your_file.csv' with your actual CSV file path
    func = 'brown'
    dcd = 'LIN'
    csv_path = f"Stats/{func}/{dcd}.csv"
    
    # Calculate statistics for all numeric columns
    # stats = calculate_statistics(csv_path)
    
    # Or specify particular columns
    stats = calculate_statistics(csv_path, columns=['n_iter_opt'])
    
    # prin the 
    print_statistics(stats)