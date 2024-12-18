import os
import pandas as pd
import psycopg2
from sqlalchemy import create_engine

# Path to the folder containing CSV files
folder_path = r"D:\TVS\shakthi\anime"

# PostgreSQL connection details
db_host = "localhost" 
db_port = "5432" # Replace with your PostgreSQL host
db_name = "anime_db"  # Replace with your database name
db_user = "postgres"  # Replace with your PostgreSQL username
db_password = "shakthi"  # Replace with your PostgreSQL password

# Create a PostgreSQL connection string
conn_str = f'postgresql+psycopg2://{db_user}:{db_password}@{db_host}/{db_name}'

# Create a SQLAlchemy engine
engine = create_engine(conn_str)

# Iterate through each CSV file in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv"):
        file_path = os.path.join(folder_path, file_name)
        
        # Read CSV file into a pandas DataFrame
        table_name = os.path.splitext(file_name)[0]  # Use file name (without extension) as table name
        try:
            df = pd.read_csv(file_path)
            # Store the DataFrame in PostgreSQL (replace existing table if it exists)
            df.to_sql(table_name, engine, if_exists='replace', index=False)
            print(f"Table '{table_name}' created successfully in the database.")
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

print(f"All CSVs have been imported into the PostgreSQL database '{db_name}'.")
