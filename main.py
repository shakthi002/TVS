import os
import google.generativeai as genai
import pandas as pd
import streamlit as st
import speech_recognition as sr  # Import SpeechRecognition
from db_operations import extract_tables_and_columns, get_table_schema, execute_sql_query
from nlp_utils import extract_keywords_and_entities, precompute_schema_embeddings, find_relevant_tables

# Load environment variables
API_KEY = "AIzaSyA61I0rexXHdbAObWjlHWTaaLZLY1zQM7k"
if not API_KEY:
    st.error("API key not found. Please set it in your environment variables.")
else:
    genai.configure(api_key=API_KEY)

# Function to generate a dynamic prompt for SQL query generation based on relevant tables
def generate_dynamic_prompt(db_paths, question, relevant_tables):
    prompt = "You are an expert in converting English questions to SQL queries.\n\n"
    prompt += "The SQL database contains the following relevant tables:\n"
    for db_path in db_paths:
        for table in relevant_tables:
            columns = get_table_schema(db_path, table)
            if isinstance(columns, str):  # Error occurred
                return f"Error fetching columns for table {table}: {columns}"
            if columns:
                prompt += f"\nTable: {table}\nColumns: {', '.join(columns)}\n"
    prompt += f"\nQuestion: '{question}'\nSQL:"
    return prompt

# Function to handle voice input
def get_voice_input():
    # Initialize recognizer
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        st.info("Listening for your voice input...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        # Recognize speech using Google Speech API
        voice_text = recognizer.recognize_google(audio)
        st.success(f"Recognized: {voice_text}")
        return voice_text
    except sr.UnknownValueError:
        st.error("Sorry, I could not understand the audio.")
        return ""
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Speech service; {e}")
        return ""

# Main Execution: Integrating both functionalities
def main():
    st.set_page_config(page_title="SQL Query Generator and Execution")
    st.title("SQL Query Generator and Execution")
    st.write("Generate SQL queries dynamically based on your database schema and a natural language question.")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Button for voice input
    if st.button("Speak your SQL Question"):
        # Get voice input and process
        question = get_voice_input()

        if question:
            # Append user's question to the chat history
            st.session_state.messages.append({"role": "user", "content": question})

            # Start processing the question
            db_paths = ["db1.db", "db2.db", "db3.db"]  # Example DB paths
            missing_dbs = [db for db in db_paths if not os.path.exists(db)]
            if missing_dbs:
                error_message = f"The following database files do not exist: {', '.join(missing_dbs)}"
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                st.error(error_message)
                return

            try:
                # Extract keywords and entities
                keywords, entities = extract_keywords_and_entities(question)

                # Precompute schema embeddings for all databases
                schema_embeddings = precompute_schema_embeddings(db_paths)

                # Find relevant tables based on the query
                relevant_tables = find_relevant_tables(schema_embeddings, keywords, entities)

                # Generate the dynamic prompt for the SQL query
                dynamic_prompt = generate_dynamic_prompt(db_paths, question, relevant_tables)
                if dynamic_prompt.startswith("Error"):
                    st.session_state.messages.append({"role": "assistant", "content": dynamic_prompt})
                    st.error(dynamic_prompt)
                    return

                # Generate SQL query using Gemini API
                combined_prompt = "\n".join([msg["content"] for msg in st.session_state.messages if msg["role"] == "user"]) + "\n" + dynamic_prompt
                response = genai.GenerativeModel("gemini-pro").generate_content([combined_prompt])

                if response and hasattr(response, "text"):
                    generated_sql = response.text.strip()

                    # Execute the SQL query
                    results, columns = execute_sql_query(generated_sql, db_paths)  # Using the first DB as an example

                    # Format the results and columns for display as a table
                    if isinstance(results, list) and results:
                        # Create a DataFrame for better table formatting
                        df = pd.DataFrame(results, columns=columns)

                        # Convert the DataFrame to markdown table for chat display
                        table_markdown = df.to_markdown(index=False)

                        # Append the result to the chat history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Query Results:\n\n{table_markdown}"
                        })

                        # Display the results as a dataframe in Streamlit
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

# Run the main function
if __name__ == "__main__":
    main()
