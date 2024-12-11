# import os
# import cohere
# from langchain_pinecone import PineconeVectorStore
# from langchain.chains import ConversationalRetrievalChain
# from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
# from langchain.embeddings.cohere import CohereEmbeddings
# from langchain.llms.cohere import Cohere

# os.environ["COHERE_API_KEY"] = "V5jqL0gNm1xo9gIzN0x32c9Nc814b4gh9gpOY6EY"
# os.environ['PINECONE_API_KEY']="pcsk_22X17E_TyjHKBUTtPwKSdfG7dMkcnCGHqPBShmgd9cugqxdFmVuBXYxWrnEgXDV4id7Stq"

# # Initialize Cohere client
# cohere_client = cohere.Client(api_key=os.environ["COHERE_API_KEY"])
# embeddings = CohereEmbeddings(model="embed-english-v3.0")

# # Define system and human templates
# system_template = "System message template: {context}"
# human_template = "Human message template: {question}"

# # Create message prompts
# messages = [
#     SystemMessagePromptTemplate.from_template(system_template),
#     HumanMessagePromptTemplate.from_template(human_template)
# ]

# # Create chat prompt template
# chat_prompt = ChatPromptTemplate.from_messages(messages)

# # Use the Cohere client directly as the LLM
# llm = Cohere(cohere_api_key=os.environ["COHERE_API_KEY"])

# # Initialize Cohere embeddings
# # embeddings = CohereEmbeddings(model="embed-english-v3.0")

# # Create Pinecone Vector Store retriever
# index_name = "tvs"
# retriever = PineconeVectorStore.from_existing_index(index_name, embeddings).as_retriever()
# # Create ConversationalRetrievalChain
# qa_chain = ConversationalRetrievalChain.from_llm(
#     llm=llm,
#     retriever=retriever,
#     combine_docs_chain_kwargs={"prompt": chat_prompt},
#     chain_type="stuff"  # Add chain type for Cohere
# )

# # Example usage

# chathistory = []

# # search_results = retriever.search(query)
# # print(search_results)
# response = qa_chain.run({"question": query,"chat_history":""})
# print(response)

