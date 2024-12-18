import psycopg2
from psycopg2 import sql

# Function to extract tables and columns from the database
def extract_tables_and_columns(dbpath):
    """
    Extracts all tables and their columns from the PostgreSQL database.

    Args:
        dbpath (str): Connection string for the PostgreSQL database.
    
    Returns:
        dict: Dictionary with table names as keys and column names as values.
    """
    try:
        conn = psycopg2.connect(dbpath)
        cursor = conn.cursor()
        
        # Get all table names in the 'public' schema
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public';
        """)
        tables = cursor.fetchall()

        table_columns = {}

        for table in tables:
            table_name = table[0]
            cursor.execute(sql.SQL("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = %s;
            """), [table_name])
            columns = cursor.fetchall()
            table_columns[table_name] = [col[0] for col in columns]  # Store column names

        conn.close()
        return table_columns

    except psycopg2.Error as e:
        return f"Error extracting tables and columns: {e}"

# Function to retrieve table schema (columns) from the database
def get_table_schema(db, table_name):
    """
    Retrieves the column names for a specific table in the PostgreSQL database.

    Args:
        db (str): Connection string for the PostgreSQL database.
        table_name (str): Name of the table to retrieve schema for.
    
    Returns:
        list: List of column names for the given table.
    """
    try:
        conn = psycopg2.connect(db)
        cursor = conn.cursor()
        cursor.execute(sql.SQL("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = %s;
        """), [table_name])
        columns = cursor.fetchall()
        conn.close()
        return [col[0] for col in columns]
    except psycopg2.Error as e:
        return f"Error fetching schema for table {table_name}: {e}"

# Function to execute the SQL query
def execute_sql_query(query, db_path):
    """
    Executes the given SQL query on the first PostgreSQL database in the list of database paths.

    Args:
        query (str): The SQL query to execute.
        db_paths (list): List of PostgreSQL connection strings.
    
    Returns:
        tuple: (Rows from the query, column names) or error message.
    """
    db = db_path  # Use the first database connection string in the list
    try:
        # Thoroughly clean the query to remove unwanted formatting
        cleaned_query = query.replace("```sql", "").replace("```", "").strip()
        
        # Ensure the query only contains valid SQL and no markdown artifacts
        if "```" in cleaned_query:
            raise ValueError("Query contains invalid markdown artifacts.")
        
        conn = psycopg2.connect(db)
        cursor = conn.cursor()

        # Execute the cleaned query
        cursor.execute(cleaned_query)
        rows = cursor.fetchall()

        # Fetch column names dynamically
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        
        conn.commit()
        conn.close()

        return rows, columns
    except psycopg2.Error as e:
        return f"Error executing query: {e}", []
    except ValueError as ve:
        return f"Error: {ve}", []