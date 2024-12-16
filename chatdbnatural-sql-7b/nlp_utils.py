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

# Precompute schema embeddings for tables and columns
def precompute_schema_embeddings(dbpaths):
    schema_embeddings = {}
    for dbpath in dbpaths:
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