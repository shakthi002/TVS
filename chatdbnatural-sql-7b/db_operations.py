# db_operations.py
import sqlite3

# Function to extract tables and columns from the database
def extract_tables_and_columns(dbpath):
    conn = sqlite3.connect(dbpath)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    table_columns = {}

    for table in tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        table_columns[table_name] = [col[1] for col in columns]  # Store column names

    conn.close()
    return table_columns

# Function to retrieve table schema (columns) from the database
def get_table_schema(db, table_name):
    try:
        conn = sqlite3.connect(db)
        cur = conn.cursor()
        cur.execute(f"PRAGMA table_info({table_name});")
        columns = cur.fetchall()
        conn.close()
        return [col[1] for col in columns]
    except sqlite3.Error as e:
        return f"Error fetching schema for table {table_name}: {e}"

# Function to execute the SQL query
def execute_sql_query(query, db_paths):
    try:
        # Thoroughly clean the query to remove unwanted formatting
        cleaned_query = query.replace("```sql", "").replace("```", "").strip()
        
        # Ensure the query only contains valid SQL and no markdown artifacts
        if "```" in cleaned_query:
            raise ValueError("Query contains invalid markdown artifacts.")
        
        # Loop through the provided database paths
        for db in db_paths:
            try:
                # Connect to the current database
                conn = sqlite3.connect(db)
                cur = conn.cursor()
                
                # Execute the cleaned query
                cur.execute(cleaned_query)
                rows = cur.fetchall()
                
                # Fetch column names dynamically
                columns = [desc[0] for desc in cur.description] if cur.description else []
                
                # Commit changes and close the connection
                conn.commit()
                conn.close()

                # If rows and columns are found, return them
                if rows and columns:
                    return rows, columns

            except sqlite3.Error as e:
                # Log the error for the current database and continue to the next
                print(f"Error executing query on {db}: {e}")

        # If the loop completes and no results were found
        return "No relevant tables or data found across all databases.", []

    except ValueError as ve:
        return f"Error: {ve}", []
