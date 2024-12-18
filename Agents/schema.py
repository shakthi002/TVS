import sqlite3

def generate_create_statements(db_file):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Query to fetch all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    # Iterate through each table
    for table in tables:
        table_name = table[0]
        print(f"-- Table: {table_name}")
        
        # Query to fetch the schema of the table
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()

        # Start the CREATE TABLE statement
        create_statement = f"CREATE TABLE IF NOT EXISTS {table_name} (\n"

        # Add each column to the CREATE TABLE statement
        for column in columns:
            column_name = column[1]
            column_type = column[2]
            is_not_null = "NOT NULL" if column[3] else ""
            default_value = f"DEFAULT {column[4]}" if column[4] else ""
            column_definition = f"    {column_name} {column_type} {is_not_null} {default_value}".strip()
            create_statement += column_definition + ",\n"

        # Remove the trailing comma and add closing parenthesis
        create_statement = create_statement.rstrip(",\n") + "\n);"
        print(create_statement)
        print()

    # Close the connection
    conn.close()

# Call the function with your database file
db_file = 'your_database.db'  # Replace with your SQLite database file
generate_create_statements("anime.db")
