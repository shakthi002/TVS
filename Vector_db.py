# from langchain_cohere import CohereEmbeddings
# from langchain_pinecone import PineconeVectorStore
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# import os

# os.environ["COHERE_API_KEY"] = "V5jqL0gNm1xo9gIzN0x32c9Nc814b4gh9gpOY6EY"
# os.environ['PINECONE_API_KEY']="pcsk_22X17E_TyjHKBUTtPwKSdfG7dMkcnCGHqPBShmgd9cugqxdFmVuBXYxWrnEgXDV4id7Stq"

# embeddings = CohereEmbeddings(model="embed-english-v3.0")

# combined_text = ','.join(docs)  # Combine all text into a single string

# # Initialize text splitter to create chunks
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1500,  # Adjust size according to your needs
#     chunk_overlap=200
# )
# chunks = text_splitter.split_text(combined_text)
# index_name = "logs"
# PineconeVectorStore.from_texts(
#     texts=chunks,
#     embedding=embeddings,
#     index_name=index_name
# )
