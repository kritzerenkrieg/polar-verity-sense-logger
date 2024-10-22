import os
import pandas as pd
import matplotlib.pyplot as plt

# Function to list CSV files in the current directory
def list_csv_files():
    files = [f for f in os.listdir('.') if f.endswith('.csv')]
    return files

# Function to read the CSV file and filter heart rate measurements
def process_heart_rate_data(file_path):
    # Read the CSV file
    data = pd.read_csv(file_path, header=None, names=["timestamp", "sender", "data"])
    
    # Filter only heart rate measurements
    heart_rate_data = data[data['data'].str.contains('Heart Rate Measurement')]
    
    # Extract the timestamp and heart rate values
    heart_rate_data['heart_rate'] = heart_rate_data['data'].str.extract(r"bytearray\(b'\\x00(\w)'\)")[0]
    
    # Convert heart rate to integer (ASCII to decimal)
    heart_rate_data['heart_rate'] = heart_rate_data['heart_rate'].apply(lambda x: ord(x) if pd.notna(x) else None)
    
    # Convert timestamp to datetime
    heart_rate_data['timestamp'] = pd.to_datetime(heart_rate_data['timestamp'])
    
    return heart_rate_data[['timestamp', 'heart_rate']]

# Function to plot heart rate measurements
def plot_heart_rate(heart_rate_data):
    plt.figure(figsize=(12, 6))
    plt.plot(heart_rate_data['timestamp'], heart_rate_data['heart_rate'], marker='o', linestyle='-', color='b')
    plt.title('Heart Rate Measurements Over Time')
    plt.xlabel('Time')
    plt.ylabel('Heart Rate (BPM)')
    plt.xticks(rotation=45)
    plt.grid()
    plt.tight_layout()
    plt.show()

# Main function
def main():
    # List CSV files in the current directory
    csv_files = list_csv_files()
    
    # Check if any CSV files were found
    if not csv_files:
        print("No CSV files found in the current directory.")
        return
    
    # Display available files to the user
    print("Available CSV files:")
    for i, file in enumerate(csv_files, start=1):
        print(f"{i}. {file}")
    
    # Get user choice
    choice = input("Select a CSV file by entering the number (1 - {}): ".format(len(csv_files)))
    
    try:
        choice_index = int(choice) - 1
        if choice_index < 0 or choice_index >= len(csv_files):
            raise ValueError("Invalid selection.")
    except ValueError:
        print("Invalid input. Please enter a valid number.")
        return
    
    # Process the selected CSV file
    selected_file = csv_files[choice_index]
    heart_rate_data = process_heart_rate_data(selected_file)
    
    # Print the heart rate measurements
    print(heart_rate_data)
    
    # Plot the heart rate measurements
    plot_heart_rate(heart_rate_data)

# Execute the main function
if __name__ == "__main__":
    main()
