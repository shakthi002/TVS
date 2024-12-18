# nlp_utils.py
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
from db_operations import extract_tables_and_columns  # Import from db_operations.py

# Load SpaCy model
nlp = spacy.load('en_core_web_lg')

# Cache word vector computations
@lru_cache(maxsize=None)
def get_word_vector(word):
    return nlp(word).vector

def generate_dynamic_prompt(db_path, question, relevant_tables):
    prompt = (
        "You are an expert in converting English questions to SQL queries. Please follow these SQL rules for Postgres:\n"
        "- Use 'ILIKE' for case-insensitive text filtering (e.g., to match partial text).\n"
        "- Use only the available tables and columns that exist in the provided schema.\n"
        "- Avoid calculations like age; if such data is required, it must exist in the database directly.\n"
        "- Ensure the SQL query is syntactically correct and efficient for PostgreSQL.\n\n"
    )
    
    prompt += "The SQL database contains the following relevant tables and their columns:\n"
    
    # Adding schema directly into the prompt
    prompt += """
    -- Table: animes
    CREATE TABLE IF NOT EXISTS animes (
        uid INTEGER PRIMARY KEY,
        title TEXT,
        synopsis TEXT,
        genre TEXT,
        aired DATE,
        episodes REAL,
        members INTEGER,
        popularity INTEGER,
        ranked REAL,
        score REAL,
        img_url TEXT,
        link TEXT
    );

    -- Table: profiles
    CREATE TABLE IF NOT EXISTS profiles (
        profile TEXT PRIMARY KEY,
        gender TEXT,
        birthday DATE,
        favorites_anime TEXT,
        link TEXT
    );

    -- Table: reviews
    CREATE TABLE IF NOT EXISTS reviews (
        uid SERIAL PRIMARY KEY,
        profile TEXT REFERENCES profiles(profile),
        anime_uid INTEGER REFERENCES animes(uid),
        text TEXT,
        score INTEGER,
        scores TEXT,
        link TEXT
    );
    """
    
    prompt += f"\nQuestion: '{question}'\n"
    prompt += "Relevant Tables: " + ", ".join(relevant_tables) + "\n"
    prompt += (
        "SQL (use only the relevant tables and columns, ensuring correctness and efficiency in Postgres syntax):\n"
    )
    
    return prompt

# Precompute schema embeddings for tables and columns
def precompute_schema_embeddings(dbpath):
    schema_embeddings = {}
    # for dbpath in dbpaths:
    table_columns = extract_tables_and_columns(dbpath)
    # print(table_columns)
    for table, columns in table_columns.items():
        schema_embeddings[table] = nlp(table).vector
        for column in columns:
            schema_embeddings[f"{table}.{column}"] = nlp(column).vector
        # print(schema_embeddings)
    return schema_embeddings

# Cosine similarity computation
def compute_similarity(vec1, vec2):
    return cosine_similarity([vec1], [vec2])[0][0]

# Function to find relevant tables based on similarity
def find_relevant_tables(schema_embeddings, keywords, entities):
    relevant_tables = set()
    keyword_vectors = [get_word_vector(kw) for kw in keywords + list(entities.keys())]

    for schema_item, embedding in schema_embeddings.items():
        for keyword_vector in keyword_vectors:
            # print(compute_similarity(embedding, keyword_vector),schema_item)
            if compute_similarity(embedding, keyword_vector) > -1:
                table_name = schema_item.split(".")[0]  # Extract the table name
                relevant_tables.add(table_name)

    return list(relevant_tables)

# Extract keywords and entities from the query using SpaCy
def extract_keywords_and_entities(query):
    doc = nlp(query)
    keywords = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    entities = {ent.text: ent.label_ for ent in doc.ents}
    return keywords, entities