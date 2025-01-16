import os
import streamlit as st
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings.cohere import CohereEmbeddings
from typing import List, Tuple, Dict

# Set API Keys
os.environ["COHERE_API_KEY"] = "your_cohere_api_key"
os.environ["PINECONE_API_KEY"] = "your_pinecone_api_key"

class ChatHistory:
    """Manages chat history with proper context window management."""
    def __init__(self, max_history: int = 5):
        self.max_history = max_history
        self.messages: List[Dict[str, str]] = []
        self.conversation_pairs: List[Tuple[str, str]] = []
    
    def add_message(self, role: str, content: str):
        """Add a message to the history with proper formatting."""
        self.messages.append({"role": role, "content": content})
        
        # Update conversation pairs for the LLM context
        if role == "user":
            self._current_query = content
        elif role == "assistant" and hasattr(self, '_current_query'):
            self.conversation_pairs.append((self._current_query, content))
            self._maintain_history_window()
    
    def _maintain_history_window(self):
        """Maintain the conversation history within the specified window."""
        if len(self.conversation_pairs) > self.max_history:
            self.conversation_pairs = self.conversation_pairs[-self.max_history:]
    
    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages for display."""
        return self.messages
    
    def get_conversation_pairs(self) -> List[Tuple[str, str]]:
        """Get conversation pairs for LLM context."""
        return self.conversation_pairs

def initialize_embeddings() -> CohereEmbeddings:
    """Initialize Cohere embeddings with improved configuration."""
    return CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=os.environ["COHERE_API_KEY"]
    )

def create_chat_prompt() -> ChatPromptTemplate:
    """Create a default chat prompt template."""
    context = """You are a helpful AI assistant. Your role is to:

1. Understand user questions and requirements clearly
2. Provide detailed, relevant information
3. Be concise yet comprehensive
4. Use clear formatting for better readability

Guidelines:
- Ask clarifying questions when needed
- Provide specific examples when relevant
- Format responses clearly using markdown
- Keep responses focused and on-topic
- Use simple, accessible language

Remember to:
- Be friendly and professional
- Stay factual and accurate
- Respect user privacy
- Admit when you don't know something
"""
    
    system_template = """Use the following context for responses:
    {context}

    Chat History Reference: {chat_history}

    Guidelines:
    - Provide responses based on the current query and context
    - Reference chat history when relevant
    - Focus on the current user's needs
    - Be clear and concise
    """
    
    human_template = """User Query: {question}
    
    Please provide a response considering the context and previous conversation."""
    
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template)
    ]
    
    return ChatPromptTemplate.from_messages(messages)

def initialize_retrieval_chain() -> ConversationalRetrievalChain:
    """Initialize the conversational retrieval chain with improved configuration."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key="your_google_api_key",
        temperature=0.3,
        max_tokens=1524,
        timeout=45,
        max_retries=3,
    )
    
    embeddings = initialize_embeddings()
    retriever = PineconeVectorStore.from_existing_index(
        "your_index_name",
        embeddings
    ).as_retriever(
        search_kwargs={"k": 5}
    )
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": create_chat_prompt()},
        chain_type="stuff",
        return_source_documents=False,
        verbose=True
    )

def run_query(qa_chain: ConversationalRetrievalChain, query: str, chat_history: List[Tuple[str, str]]) -> str:
    """Run a query through the QA chain with error handling."""
    try:
        result = qa_chain({
            "question": query, 
            "chat_history": chat_history
        })
        return result["answer"]
    except Exception as e:
        raise Exception(f"Error processing query: {str(e)}")

def main():
    st.set_page_config(
        page_title="AI Assistant",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("💬 AI Assistant")
    
    st.markdown("""
    Welcome! I'm your AI assistant. I can help you:
    - Answer your questions
    - Provide detailed information
    - Offer suggestions and recommendations
    - Assist with various topics
    
    How can I help you today?
    """)
    
    # Initialize session state
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = ChatHistory(max_history=5)

    # Initialize chain if not already initialized
    if st.session_state.qa_chain is None:
        with st.spinner("Initializing AI assistant..."):
            try:
                st.session_state.qa_chain = initialize_retrieval_chain()
                st.success("Ready to help!")
            except Exception as e:
                st.error(f"Error initializing system: {e}")
                return

    # Display chat messages
    for message in st.session_state.chat_history.get_messages():
        with st.chat_message(message["role"], avatar="💬" if message["role"] == "assistant" else None):
            st.write(message["content"])

    # Chat input
    query = st.chat_input("Ask me anything...")
    
    if query:
        # Add user message
        st.session_state.chat_history.add_message("user", query)
        with st.chat_message("user"):
            st.write(query)
        
        # Generate and display response
        with st.chat_message("assistant", avatar="💬"):
            with st.spinner("Thinking..."):
                try:
                    response = run_query(
                        st.session_state.qa_chain,
                        query,
                        st.session_state.chat_history.get_conversation_pairs()
                    )
                    st.write(response)
                    st.session_state.chat_history.add_message("assistant", response)

                except Exception as e:
                    error_message = f"I apologize, but I encountered an error: {e}. Please try asking your question differently."
                    st.error(error_message)
                    st.session_state.chat_history.add_message("assistant", error_message)

if __name__ == "__main__":
    main()
