from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.llms.cohere import Cohere
# Updated Imports
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.llms import Cohere


def initialize_retrieval_chain(index_name, embeddings):
    """
    Initialize a conversational retrieval chain.
    
    Args:
        index_name (str): Name of the Pinecone index.
        embeddings: Embedding model to retrieve text.
    
    Returns:
        ConversationalRetrievalChain: Retrieval chain instance.
    """
    # Define system and human templates
    system_template = "System message template: {context}"
    human_template = "Human message template: {question}"

    # Create message prompts
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template)
    ]
    chat_prompt = ChatPromptTemplate.from_messages(messages)

    # Initialize LLM
    llm = Cohere(cohere_api_key="V5jqL0gNm1xo9gIzN0x32c9Nc814b4gh9gpOY6EY")

    # Create retriever
    retriever = PineconeVectorStore.from_existing_index(index_name, embeddings).as_retriever()

    # Create and return conversational retrieval chain
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": chat_prompt},
        chain_type="stuff"  # Define the chain type
    )


def run_query(qa_chain, query, chat_history=""):
    """
    Run a query through the retrieval chain.
    
    Args:
        qa_chain: The conversational retrieval chain.
        query (str): The query string.
        chat_history (str): Chat history for the conversation.
    
    Returns:
        str: Response from the retrieval chain.
    """
    return qa_chain.run({"question": query, "chat_history": chat_history})
