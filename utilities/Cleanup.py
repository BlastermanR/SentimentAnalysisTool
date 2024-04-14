import argparse
import pandas as pd

def keep_specified_columns(input_file, output_file, specified_columns):
    """
    Reads a CSV file, keeps only the specified columns, and saves to a new file.
    
    Args:
    - input_file: The path to the input CSV file.
    - output_file: The path to save the output CSV file.
    - specified_columns: A list of column names to keep.
    """
    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(input_file)
        
        # Keep only the specified columns
        df = df[specified_columns]
        
        # Save the DataFrame to a new CSV file
        df.to_csv(output_file, index=False)
        
        print("New CSV file saved successfully!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Keep specified columns in a CSV file")
    parser.add_argument("input_file", help="Path to the input CSV file")
    parser.add_argument("output_file", help="Path to save the output CSV file")
    parser.add_argument("columns", nargs="+", help="List of column names to keep")
    args = parser.parse_args()

    keep_specified_columns(args.input_file, args.output_file, args.columns)