import os
import pandas as pd
import matplotlib.pyplot as plt

def list_csv_files():
    """List all CSV files in the current directory."""
    files = [f for f in os.listdir('.') if f.endswith('.csv')]
    return files

def process_heart_rate_data(file_path):
    """
    Process heart rate measurement data from a CSV file.
    
    Parameters:
    file_path (str): Path to the CSV file containing heart rate data
    
    Returns:
    pandas.DataFrame: Processed data with converted heart rate values
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Filter rows where sender contains "Heart Rate Measurement"
        df = df[df['sender'].str.contains('Heart Rate Measurement')]
        
        def convert_bytearray(bytearray_str):
            try:
                # Extract the bytes value from the string
                # Remove the "bytearray(" and ")" parts
                clean_str = bytearray_str.replace("bytearray(", "").replace(")", "")
                # Convert the string to bytes object
                byte_val = eval(clean_str)  # This converts the string 'b"\x00?"' to actual bytes
                # Return the second byte (index 1) which contains the heart rate
                return byte_val[1]
            except Exception as e:
                print(f"Error converting value {bytearray_str}: {str(e)}")
                return None
        
        # Apply conversion to the data column
        df['heart_rate'] = df['data'].apply(convert_bytearray)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create a clean DataFrame with relevant columns
        result_df = df[['timestamp', 'heart_rate']]
        
        # Calculate some basic statistics
        stats = {
            'Average HR': result_df['heart_rate'].mean(),
            'Max HR': result_df['heart_rate'].max(),
            'Min HR': result_df['heart_rate'].min(),
            'Duration': (result_df['timestamp'].max() - result_df['timestamp'].min()).total_seconds() / 60
        }
        
        return result_df, stats
    
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None, None

def plot_heart_rate(heart_rate_data):
    """
    Create a visualization of heart rate data over time.
    
    Parameters:
    heart_rate_data (pandas.DataFrame): Processed heart rate data
    """
    plt.figure(figsize=(12, 6))
    
    # Create the main heart rate plot
    plt.plot(heart_rate_data['timestamp'], heart_rate_data['heart_rate'], 
             marker='o', linestyle='-', color='b', label='Heart Rate')
    
    # Add a rolling average
    rolling_avg = heart_rate_data['heart_rate'].rolling(window=5).mean()
    plt.plot(heart_rate_data['timestamp'], rolling_avg, 
             color='r', linestyle='--', label='5-point Moving Average')
    
    # Customize the plot
    plt.title('Heart Rate Measurements Over Time')
    plt.xlabel('Time')
    plt.ylabel('Heart Rate (BPM)')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Add horizontal lines for min, max, and mean
    plt.axhline(y=heart_rate_data['heart_rate'].mean(), color='g', 
                linestyle=':', label='Mean HR', alpha=0.5)
    
    plt.tight_layout()

def main():
    """Main function to run the heart rate analysis program."""
    # List CSV files in the current directory
    csv_files = list_csv_files()
    
    if not csv_files:
        print("No CSV files found in the current directory.")
        return
    
    # Display available files
    print("\nAvailable CSV files:")
    for i, file in enumerate(csv_files, start=1):
        print(f"{i}. {file}")
    
    # Get user choice
    while True:
        try:
            choice = int(input(f"\nSelect a CSV file (1-{len(csv_files)}): "))
            if 1 <= choice <= len(csv_files):
                break
            print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Process the selected file
    selected_file = csv_files[choice - 1]
    print(f"\nProcessing {selected_file}...")
    
    heart_rate_data, stats = process_heart_rate_data(selected_file)
    
    if heart_rate_data is not None and not heart_rate_data.empty:
        # Display statistics
        print("\nHeart Rate Statistics:")
        print(f"Average Heart Rate: {stats['Average HR']:.1f} BPM")
        print(f"Maximum Heart Rate: {stats['Max HR']} BPM")
        print(f"Minimum Heart Rate: {stats['Min HR']} BPM")
        print(f"Recording Duration: {stats['Duration']:.1f} minutes")
        
        # Display the first few readings
        print("\nFirst few readings:")
        print(heart_rate_data.head())
        
        # Plot the data
        plot_heart_rate(heart_rate_data)
        plt.show()
        
        # Ask if user wants to save the processed data
        save = input("\nWould you like to save the processed data? (y/n): ")
        if save.lower() == 'y':
            output_file = f"processed_{selected_file}"
            heart_rate_data.to_csv(output_file, index=False)
            print(f"Data saved to {output_file}")
    else:
        print("No valid heart rate data found in the selected file.")

if __name__ == "__main__":
    main()