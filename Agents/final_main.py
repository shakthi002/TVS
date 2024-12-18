import google.generativeai as genai
import streamlit as st
import fitz  # PyMuPDF for PDF text extraction
from io import BytesIO
import pandas as pd
import re
from langchain.agents import create_openai_tools_agent
from Vector_db import (
    set_environment_variables,
    initialize_embeddings,
    split_text_into_chunks,
    create_vector_store,
    add_docs_to_vector_store,
)
from retrive_vectordb import initialize_retrieval_chain, run_query
from db_operations import extract_tables_and_columns, get_table_schema, execute_sql_query
from nlp_utils import (
    extract_keywords_and_entities,
    precompute_schema_embeddings,
    find_relevant_tables,
    generate_dynamic_prompt,
)
from langchain.agents import Tool
from langchain.agents import AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain_core.messages import AIMessage,HumanMessage
from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
import traceback
# SQL Query Tool
class SQLQueryTool:
    def __init__(self, db_path: str, api_key: str):
        self.db_path = db_path
        self.api_key = api_key

    def _call(self, query):
        """Generate SQL query from the input and return results."""
        try:
            # Handle different input types
            if isinstance(query, dict) and 'sql' in query:
                # If a pre-formed SQL query is passed
                generated_sql = query['sql']
            elif isinstance(query, str):
                # Generate SQL dynamically using NLP techniques
                # Extract keywords and entities from the query
                keywords, entities = extract_keywords_and_entities(query)

                # Precompute schema embeddings for all databases
                schema_embeddings = precompute_schema_embeddings(self.db_path)

                # Find relevant tables based on the query
                relevant_tables = find_relevant_tables(schema_embeddings, keywords, entities)

                # Generate the dynamic prompt for the SQL query
                dynamic_prompt = generate_dynamic_prompt(self.db_path, query, relevant_tables)
                if dynamic_prompt.startswith("Error"):
                    return dynamic_prompt

                # Configure the API and generate SQL query using Gemini model
                genai.configure(api_key=self.api_key)
                response = genai.GenerativeModel("gemini-1.5-flash").generate_content([dynamic_prompt])
                
                generated_sql = response.text.strip()
            else:
                return f"Unexpected query type: {type(query)}"

            # Debug print for generated SQL
            print(f"Generated SQL Query: {generated_sql}")
            generated_sql=re.search(r"sql\n(SELECT.*?;)", generated_sql, re.DOTALL).group(1)

            # Execute the SQL query
            results, columns = execute_sql_query(generated_sql, self.db_path)

            # Validate results
            if not results:
                return "No results returned from the query."
            print(results,columns)
            # Convert results to DataFrame
            try:
                import pandas as pd
                df = pd.DataFrame(results, columns=columns)
                
                # Return as markdown table or fallback to string representation
                try:
                    return df.to_markdown(index=False)
                except Exception:
                    return df.to_string(index=False)
            
            except Exception as df_error:
                # Fallback to basic string representation if DataFrame fails
                print(f"DataFrame conversion error: {df_error}")
                return "\n".join([" | ".join(map(str, row)) for row in results])

        except Exception as e:
            # Comprehensive error logging
            print(f"Error in SQL Query Tool: {e}")
            import traceback
            traceback.print_exc()
            return f"An error occurred: {e}"

# Unstructured Data Tool
class UnstructuredDataTool:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def extract_text_from_pdf(self, pdf_file):
        """Extract text from a PDF file using PyMuPDF (fitz)."""
        pdf_document = fitz.open(stream=BytesIO(pdf_file.read()), filetype="pdf")
        text = ""
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
        return text

    def _call(self, query: str) -> str:
        try:
            qa_chain = initialize_retrieval_chain(index_name="tvs", embeddings=initialize_embeddings())
            response = run_query(qa_chain, query)
            print(response)
            return response
        except Exception as e:
            print("un")
            return f"An error occurred: {e}"


# DataChoiceTool using agent
class DataChoiceTool:
    def __init__(self, sql_tool: SQLQueryTool, unstructured_tool: UnstructuredDataTool, api_key: str):
        self.sql_tool = sql_tool
        self.unstructured_tool = unstructured_tool
        self.api_key = api_key

        # Updated tool definitions with clearer use cases and descriptions
        tools = [
            Tool(
                name="sql_query_tool",
                func=self.sql_tool._call,
                description="""
                SQL Query Tool for retrieving structured data from a PostgreSQL database.
                Use this tool when:
                - Fetching or analyzing structured data
                - Performing tasks like filtering, aggregating, or ranking data
                - Handling most general-purpose queries
                Examples:
                - "What are the top 5 highest-rated anime?"
                - "Find all anime in a specific genre."
                - "Calculate average rating for anime in a database."
                """
            ),
            Tool(
                name="unstructured_data_tool",
                func=self.unstructured_tool._call,
                description="""
                Unstructured Data Tool for processing text documents and PDFs.
                Use this tool when:
                - Searching or extracting information from text-based documents or PDFs
                - Handling queries involving reasoning or unstructured text
                Examples:
                - "Summarize this PDF document."
                - "Find information about a specific anime in my documents."
                - "Extract key points from a research paper."
                """
            )
        ]

        # Initialize the LLM with improved configurations
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=self.api_key,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

        # Define the agent with a more specific prompt for better tool selection
        self.agent = create_openai_tools_agent(
            llm=llm,
            tools=tools,
            prompt=ChatPromptTemplate.from_messages([
                ("system", """
                You are an intelligent data retrieval assistant with access to two specialized tools:
                1. SQL Query Tool: Retrieves structured data from a PostgreSQL database.
                2. Unstructured Data Tool: Searches through text documents and PDFs.

                Guidelines for Tool Selection:
                - Use the SQL Query Tool for most queries involving structured data.
                - Use the Unstructured Data Tool for reasoning queries or unstructured text processing.
                - Always analyze the input query carefully and choose the most appropriate tool.
                - By default, attempt to execute the query in both tools and combine results if possible.
                - If combining results is not feasible, return the most relevant response from the appropriate tool.

                Implementation Notes:
                - The SQL Tool uses an LLM to generate SQL queries and fetch responses from a PostgreSQL database.
                - The Unstructured Data Tool retrieves information from a vector database for unstructured text.

                Your goal is to provide the most accurate and helpful response by leveraging these tools effectively.
                """),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ])
        )

        # Initialize the agent executor with verbose output for debugging
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True
        )

    def _call(self, query: str) -> str:
        try:
            if not isinstance(query, str):
                raise ValueError("Query must be a string.")
            
            # Execute the query through the agent
            response = self.agent_executor.invoke({
                "input": query,
                "chat_history": []
            })

            # Extract the result
            output = response.get('output', 'No response generated.')
            
            # Handle SQL tool results
            if isinstance(output, list) or isinstance(output, dict):  # Assuming SQL tool returns structured data
                # Format the result as a readable table
                if isinstance(output, list) and all(isinstance(row, dict) for row in output):
                    # Convert list of dicts to a DataFrame
                    try:
                        import pandas as pd
                        df = pd.DataFrame(output)
                        return df.to_string(index=False)  # Return as a formatted string
                    except ImportError:
                        return "Data:\n" + "\n".join(str(row) for row in output)

                return str(output)
            else:
                # If the output is already a string
                return output

        except Exception as e:
            return f"An error occurred while processing the query: {e}"


        
def main():
    st.title("Dynamic Document and Query Manager")

    # Initialize environment and embeddings
    set_environment_variables()
    embeddings = initialize_embeddings()

    query = st.text_input("Ask a question:")
    if st.button("Submit Query"):
        if query.strip():
            try:
                api_key = "AIzaSyA61I0rexXHdbAObWjlHWTaaLZLY1zQM7k"  # Replace with your actual API key

                # Initialize tools with required db_path and api_key
                sql_tool = SQLQueryTool(
                    db_path="dbname=anime_db user=postgres password=shakthi host=localhost port=5432",
                    api_key=api_key
                )
                unstructured_tool = UnstructuredDataTool(api_key=api_key)

                # Initialize DataChoiceTool with agent
                data_choice_tool = DataChoiceTool(
                    sql_tool=sql_tool,
                    unstructured_tool=unstructured_tool,
                    api_key=api_key
                )

                # Get the response using DataChoiceTool
                response = data_choice_tool._call(query)
                st.write("Response:")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a query.")


if __name__ == "__main__":
    main()
