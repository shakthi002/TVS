import sqlite3

# Define table schemas and data
tables_data = {
    "users": {
        "schema": """CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL
        )""",
        "data": [
            (1, "Alice", "alice@example.com"),
            (2, "Bob", "bob@example.com"),
            (3, "Charlie", "charlie@example.com"),
            # Additional rows
            (4, "David", "david@example.com"),
            (5, "Eva", "eva@example.com")
        ]
    },
    "products": {
        "schema": """CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            price REAL NOT NULL
        )""",
        "data": [
            (1, "Laptop", 1200.50),
            (2, "Smartphone", 800.00),
            (3, "Headphones", 150.75),
            # Additional rows
            (4, "Tablet", 500.00),
            (5, "Smartwatch", 250.00)
        ]
    },
    "orders": {
        "schema": """CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
            FOREIGN KEY (product_id) REFERENCES products (id) ON DELETE CASCADE
        )""",
        "data": [
            (1, 1, 2),
            (2, 2, 3),
            (3, 3, 1),
            # Additional rows
            (4, 4, 5),
            (5, 5, 4)
        ]
    },
    "employees": {
        "schema": """CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            department TEXT NOT NULL
        )""",
        "data": [
            (1, "Eve", "HR"),
            (2, "Frank", "IT"),
            (3, "Grace", "Finance"),
            # Additional rows
            (4, "Hannah", "Marketing"),
            (5, "Ian", "Sales")
        ]
    },
    "sales": {
        "schema": """CREATE TABLE IF NOT EXISTS sales (
            id INTEGER PRIMARY KEY,
            product_id INTEGER NOT NULL,
            amount INTEGER NOT NULL,
            sale_date TEXT NOT NULL,
            FOREIGN KEY (product_id) REFERENCES products (id) ON DELETE CASCADE
        )""",
        "data": [
            (1, 1, 5, "2024-12-01"),
            (2, 2, 2, "2024-12-02"),
            (3, 3, 10, "2024-12-03"),
            # Additional rows
            (4, 4, 3, "2024-12-04"),
            (5, 5, 7, "2024-12-05")
        ]
    },
    "inventory": {
        "schema": """CREATE TABLE IF NOT EXISTS inventory (
            id INTEGER PRIMARY KEY,
            product_id INTEGER NOT NULL,
            stock_quantity INTEGER NOT NULL,
            FOREIGN KEY (product_id) REFERENCES products (id) ON DELETE CASCADE
        )""",
        "data": [
            (1, 1, 50),
            (2, 2, 100),
            (3, 3, 200),
            # Additional rows
            (4, 4, 75),
            (5, 5, 150)
        ]
    },
    "customers": {
        "schema": """CREATE TABLE IF NOT EXISTS customers (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            address TEXT NOT NULL
        )""",
        "data": [
            (1, "John Doe", "123 Main St"),
            (2, "Jane Smith", "456 Oak Rd"),
            (3, "Samuel Green", "789 Pine Ln"),
            # Additional rows
            (4, "Alice Brown", "321 Elm St"),
            (5, "Bob White", "654 Maple Ave")
        ]
    },
    "transactions": {
        "schema": """CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            amount REAL NOT NULL,
            transaction_date TEXT NOT NULL,
            FOREIGN KEY (customer_id) REFERENCES customers (id) ON DELETE CASCADE
        )""",
        "data": [
            (1, 1, 1500.00, "2024-12-01"),
            (2, 2, 800.00, "2024-12-02"),
            (3, 3, 200.75, "2024-12-03"),
            # Additional rows
            (4, 4, 500.50, "2024-12-04"),
            (5, 5, 300.30, "2024-12-05")
        ]
    },
    "suppliers": {
        "schema": """CREATE TABLE IF NOT EXISTS suppliers (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            contact_info TEXT NOT NULL
        )""",
        "data": [
            (1, "Tech Supplies", "contact@techsupplies.com"),
            (2, "Gadget World", "info@gadgetworld.com"),
            (3, "Electro Goods", "support@electrogoods.com"),
            # Additional rows
            (4, "Device Depot", "sales@devicedepot.com"),
            (5, "Gizmo Solutions", "contact@gizmosolutions.com")
        ]
    }
}

# Define which tables go into which databases (with no repeated tables)
db_tables = {
    "db1.db": ["users", "products", "orders", "employees", "sales"],  # db1 with 5 unique tables
    "db2.db": ["inventory", "customers", "transactions", "suppliers"],  # db2 with 4 unique tables
    "db3.db": ["orders", "sales", "employees", "transactions"],  # db3 with 4 unique tables
}

# Function to create and populate a single database
def create_and_populate_db(db_name, tables):
    # Open a connection to the specific database
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON;")  # Enable foreign key enforcement

    print(f"Creating tables in {db_name}...")

    # Iterate over the tables in the current database
    for table in tables:
        try:
            # Check if the table exists before attempting to create it
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}';")
            if cursor.fetchone() is None:
                print(f"Creating table: {table} in {db_name}")
                cursor.execute(tables_data[table]["schema"])
            else:
                print(f"Table '{table}' already exists in {db_name}. Skipping creation.")

            # Insert data into the table
            print(f"Inserting data into {table}...")
            cursor.executemany(
                f"INSERT INTO {table} VALUES ({', '.join(['?'] * len(tables_data[table]['data'][0]))})",
                tables_data[table]["data"],
            )

            print(f"Data inserted into {table} in {db_name}")
        except sqlite3.OperationalError as e:
            print(f"Error creating or inserting into {table} in {db_name}: {e}")

    # Commit the changes and close the connection
    conn.commit()
    conn.close()
    print(f"Database '{db_name}' created with tables: {', '.join(tables)}")

# Create databases and populate them with tables and data
for db_name, tables in db_tables.items():
    create_and_populate_db(db_name, tables)