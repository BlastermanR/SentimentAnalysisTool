import argparse
import pandas as pd

def merge_csv_files(file1, file2, output_file):
    """
    Merge two CSV files and save the combined list to a new file.
    
    Args:
    - file1: Path to the first input CSV file.
    - file2: Path to the second input CSV file.
    - output_file: Path to save the output CSV file.
    """
    try:
        # Read the CSV files into pandas DataFrames
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        
        # Merge the DataFrames
        merged_df = pd.concat([df1, df2], ignore_index=True)
        
        # Save the merged DataFrame to a new CSV file
        merged_df.to_csv(output_file, index=False)
        
        print("Merged CSV file saved successfully!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge two CSV files")
    parser.add_argument("file1", help="Path to the first input CSV file")
    parser.add_argument("file2", help="Path to the second input CSV file")
    parser.add_argument("output_file", help="Path to save the output merged CSV file")
    args = parser.parse_args()

    merge_csv_files(args.file1, args.file2, args.output_file)