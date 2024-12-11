import os
import sqlite3
import google.generativeai as genai
from dotenv import load_dotenv
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import streamlit as st
from functools import lru_cache  # Add this import at the top

# Load environment variables
load_dotenv()
API_KEY = "AIzaSyA61I0rexXHdbAObWjlHWTaaLZLY1zQM7k"
if not API_KEY:
    st.error("API key not found. Please set it in your environment variables.")
else:
    genai.configure(api_key=API_KEY)

# Load SpaCy model
nlp = spacy.load('en_core_web_lg')

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

# Precompute schema embeddings for tables and columns
def precompute_schema_embeddings(dbpaths):
    schema_embeddings = {}
    for dbpath in dbpaths:
        table_columns = extract_tables_and_columns(dbpath)
        for table, columns in table_columns.items():
            schema_embeddings[table] = nlp(table).vector
            for column in columns:
                schema_embeddings[f"{table}.{column}"] = nlp(column).vector
    return schema_embeddings

# Cache word vector computations
@lru_cache(maxsize=None)
def get_word_vector(word):
    return nlp(word).vector

# Cosine similarity computation
def compute_similarity(vec1, vec2):
    return cosine_similarity([vec1], [vec2])[0][0]

# Function to find relevant tables based on similarity
def find_relevant_tables(schema_embeddings, keywords, entities):
    relevant_tables = set()
    keyword_vectors = [get_word_vector(kw) for kw in keywords + list(entities.keys())]

    for schema_item, embedding in schema_embeddings.items():
        for keyword_vector in keyword_vectors:
            if compute_similarity(embedding, keyword_vector) > 0.5:
                table_name = schema_item.split(".")[0]  # Extract the table name
                relevant_tables.add(table_name)

    return list(relevant_tables)

# Extract keywords and entities from the query using SpaCy
def extract_keywords_and_entities(query):
    doc = nlp(query)
    keywords = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    entities = {ent.text: ent.label_ for ent in doc.ents}
    return keywords, entities

# Function to generate a dynamic prompt for SQL query generation based on relevant tables
def generate_dynamic_prompt(db, question, relevant_tables):
    prompt = "You are an expert in converting English questions to SQL queries.\n\n"
    prompt += "The SQL database contains the following relevant tables:\n"

    for table in relevant_tables:
        columns = get_table_schema(db, table)
        if isinstance(columns, str):  # Error occurred
            return columns
        prompt += f"\nTable: {table}\nColumns: {', '.join(columns)}\n"

    prompt += f"\nQuestion: '{question}'\nSQL:"
    return prompt

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
def execute_sql_query(query, db):
    try:
        # Thoroughly clean the query to remove unwanted formatting
        cleaned_query = query.replace("```sql", "").replace("```", "").strip()

        # Ensure the query only contains valid SQL and no markdown artifacts
        if "```" in cleaned_query:
            raise ValueError("Query contains invalid markdown artifacts.")

        # Connect to the SQLite database
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
        return rows, columns
    except sqlite3.Error as e:
        return f"Error executing SQL query: {e}", []
    except ValueError as ve:
        return f"Error: {ve}", []

# Main Execution: Integrating both functionalities
def main():
    # Streamlit App UI
    st.set_page_config(page_title="SQL Query Generator and Execution")
    st.title("SQL Query Generator and Execution")
    st.write("Generate SQL queries dynamically based on your database schema and a natural language question.")

    # Input fields for the user
    db_paths_input = st.text_input("Enter the paths to your SQLite databases (comma-separated):", value="db1.db, db2.db")
    question = st.text_input("Enter your SQL question:", value="What is the total amount spent in the USA?")
    generate_button = st.button("Generate and Execute SQL Query")

    if generate_button:
        db_paths = db_paths_input.split(",")  # Handle multiple DB paths
        db_paths = [db_path.strip() for db_path in db_paths]

        # Check if all the provided databases exist
        missing_dbs = [db for db in db_paths if not os.path.exists(db)]
        if missing_dbs:
            st.error(f"The following database files do not exist: {', '.join(missing_dbs)}")
        else:
            # Extract relevant tables and columns from all the provided databases
            keywords, entities = extract_keywords_and_entities(question)

            # Precompute schema embeddings once for all databases
            schema_embeddings = precompute_schema_embeddings(db_paths)

            # Find relevant tables based on the query
            relevant_tables = find_relevant_tables(schema_embeddings, keywords, entities)

            st.subheader("Relevant Tables:")
            st.write(relevant_tables)

            # Generate SQL query dynamically based only on relevant tables
            dynamic_prompt = generate_dynamic_prompt(db_paths[0], question, relevant_tables)
            if isinstance(dynamic_prompt, str) and dynamic_prompt.startswith("Error"):
                st.error(dynamic_prompt)
            else:
                st.subheader("Generated Prompt:")
                st.code(dynamic_prompt, language="plaintext")

                # Ask Gemini API for the SQL query based on the prompt and user input
                try:
                    response = genai.GenerativeModel("gemini-pro").generate_content([dynamic_prompt])
                    if response and hasattr(response, "text"):
                        generated_sql = response.text.strip()
                        st.subheader("Generated SQL Query:")
                        st.code(generated_sql, language="sql")

                        # Execute the SQL query on the first database (you can loop through databases if needed)
                        results, columns = execute_sql_query(generated_sql, db_paths[0])
                        if isinstance(results, list):
                            if results:
                                st.subheader("Query Results:")
                                df = pd.DataFrame(results, columns=columns)
                                st.dataframe(df)
                            else:
                                st.info("No results returned from the query.")
                        else:
                            st.error(results)
                    else:
                        st.error("Failed to generate SQL query. Check API configuration.")
                except Exception as e:
                    st.error(f"Error communicating with Gemini API: {e}")

# Run the main function
if __name__ == "__main__":
    main()