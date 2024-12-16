import os
import google.generativeai as genai
import pandas as pd
import streamlit as st
from db_operations import extract_tables_and_columns, get_table_schema, execute_sql_query
from nlp_utils import extract_keywords_and_entities, precompute_schema_embeddings, find_relevant_tables
from transformers import AutoTokenizer, AutoModelForCausalLM
import sqlite3

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("chatdb/natural-sql-7b")
model = AutoModelForCausalLM.from_pretrained("chatdb/natural-sql-7b")

# Load environment variables
# API_KEY = "AIzaSyA61I0rexXHdbAObWjlHWTaaLZLY1zQM7k"
# if not API_KEY:
#     st.error("API key not found. Please set it in your environment variables.")
# else:
#     genai.configure(api_key=API_KEY)

def get_all_tables(db_path):
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
        return tables
    except Exception as e:
        return f"Error fetching tables: {str(e)}"

def generate_dynamic_prompt(db_paths, question, relevant_tables):
    prompt = (
        "You are an expert in converting English questions into efficient SQL queries for SQLite.\n\n"
        "The SQL database contains the following tables with their columns:\n"
    )

    table_details = {}

    for db_path in db_paths:
        all_tables = get_all_tables(db_path)
        if isinstance(all_tables, str):
            return f"Error fetching tables: {all_tables}"

        for table in all_tables:
            columns = get_table_schema(db_path, table)
            if isinstance(columns, str):
                return f"Error fetching columns for table {table}: {columns}"
            if columns:
                table_details[table] = columns

    for table, columns in table_details.items():
        prompt += f"\nTable: {table}\nColumns: {', '.join(columns)}\n"

    prompt += f"\nThe relevant tables for this question are: {', '.join(relevant_tables)}\n"
    prompt += (
        "\nMake sure to follow these guidelines when constructing the SQL query:\n"
        "1. Use LIKE for text-based filters (e.g., WHERE column_name LIKE '%value%').\n"
        "2. Use INNER JOIN or LEFT JOIN as needed when querying across multiple tables.\n"
        "3. Only use the tables and columns defined in the provided schema.\n"
        "4. Ensure the query is syntactically correct and efficient.\n"
        "5. If the question asks for aggregate data (e.g., sum, average), use GROUP BY and appropriate aggregate functions.\n"
    )

    prompt += f"\nQuestion: '{question}'\nSQL Query:"
    return prompt

def main():
    st.set_page_config(page_title="SQL Query Generator and Execution")
    st.title("SQL Query Generator and Execution")
    st.write("Generate SQL queries dynamically based on your database schema and a natural language question.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Enter your SQL question:")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})

        db_paths = ["db1.db"]
        missing_dbs = [db for db in db_paths if not os.path.exists(db)]
        if missing_dbs:
            error_message = f"The following database files do not exist: {', '.join(missing_dbs)}"
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            st.error(error_message)
            return

        try:
            keywords, entities = extract_keywords_and_entities(question)
            schema_embeddings = precompute_schema_embeddings(db_paths)
            relevant_tables = find_relevant_tables(schema_embeddings, keywords, entities)
            dynamic_prompt = generate_dynamic_prompt(db_paths, question, relevant_tables)

            if dynamic_prompt.startswith("Error"):
                st.session_state.messages.append({"role": "assistant", "content": dynamic_prompt})
                st.error(dynamic_prompt)
                return

            combined_prompt = "\n".join([msg["content"] for msg in st.session_state.messages if msg["role"] == "user"]) + "\n" + dynamic_prompt
            inputs = tokenizer(combined_prompt, return_tensors="pt")

            outputs = model.generate(
                **inputs, 
                max_length=150,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7
            )

            if outputs is not None:
                generated_sql = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
                results, columns = execute_sql_query(generated_sql, db_paths)

                if isinstance(results, list) and results:
                    df = pd.DataFrame(results, columns=columns)
                    table_markdown = df.to_markdown(index=False)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Query Results:\n\n{table_markdown}"
                    })

                    with st.chat_message("user"):
                        st.markdown(question)

                    with st.chat_message("assistant"):
                        st.markdown("Query Results:")
                    st.dataframe(df)
                else:
                    st.session_state.messages.append({"role": "assistant", "content": "No results returned from the query."})
                    st.info("No results returned from the query.")
            else:
                error_message = "Failed to generate SQL query. Check API configuration or input."
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                st.error(error_message)

        except Exception as e:
            error_message = f"An error occurred: {e}"
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            st.error(error_message)

if __name__ == "__main__":
    main()