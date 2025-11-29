import pandas as pd
import os

def clean_csv_file():
    # Configuration
    file_path = r'D:\nature_everything\live_rating.csv'
    output_path = r'D:\nature_everything\live_rating_cleaned.csv'

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        return

    print(f"Reading file: {file_path}...")

    try:
        # 1. Read the CSV
        # skipinitialspace=True helps if there are spaces after commas in the header (e.g. "Patient, Activity, Rating")
        df = pd.read_csv(file_path, skipinitialspace=True)

        # 2. Clean Column Names
        # This removes any accidental leading/trailing spaces from the headers
        df.columns = df.columns.str.strip()

        # Verify 'Rating' column exists
        if 'Rating' not in df.columns:
            print(f"Error: Column 'Rating' not found. Available columns: {list(df.columns)}")
            return

        # 3. Filter Data
        # This removes standard empty values (NaN) AND specific strings like "N/A"
        original_count = len(df)
        
        # Convert column to numeric, coercing errors to NaN (this turns 'N/A' strings into NaN)
        df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
        
        # Drop rows with NaN in 'Rating'
        df_cleaned = df.dropna(subset=['Rating'])

        # OPTIONAL: Convert Rating back to integer if you don't want decimals (e.g., 2.0 -> 2)
        df_cleaned['Rating'] = df_cleaned['Rating'].astype(int)

        cleaned_count = len(df_cleaned)
        removed_count = original_count - cleaned_count

        # 4. Save the file
        df_cleaned.to_csv(output_path, index=False)

        print("------------------------------------------------")
        print("Processing Complete.")
        print(f"Original rows: {original_count}")
        print(f"Rows removed:  {removed_count}")
        print(f"Remaining rows: {cleaned_count}")
        print(f"File saved to: {output_path}")
        print("------------------------------------------------")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    clean_csv_file()