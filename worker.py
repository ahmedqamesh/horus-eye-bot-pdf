import torch
import logging
# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
from langchain_core.prompts import PromptTemplate  # Updated import per deprecation notice
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader  # New import path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma  # New import path
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import warnings
warnings.filterwarnings("ignore", message="Can't initialize NVML")

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Check for GPU availability and set the appropriate device for computation.
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Global variables
conversation_retrieval_chain = None
chat_history = []
llm_hub = None
embeddings = None

# Initialize local LLaMA LLM
def init_llm(model_id: str):
    global llm_hub, embeddings

    logger.info("Initializing LLM and embeddings...")

     # Local LLaMA 2 model
    #MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"  # free for research/commercial
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",   # automatic GPU assignment if available
        torch_dtype=torch.float16,  # reduce memory usage
        low_cpu_mem_usage=True
    )

    # HuggingFace pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.1,
    )

    # Wrap with LangChain
    llm_hub = HuggingFacePipeline(pipeline=pipe)
    logger.info("Local LLaMA2 model initialized successfully.")
    # Initialize embeddings for document retrieval
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": DEVICE}
    )

    logger.debug("Embeddings initialized with model device: %s", DEVICE)
    logger.info("LLM and embeddings initialization complete.")
    return llm_hub

# Function to process a PDF document
def process_document(document_path):
    global conversation_retrieval_chain, embeddings
    logger.info("Loading document from path: %s", document_path)
    # Load the document
    loader = PyPDFLoader(document_path)
    documents = loader.load()
    if not documents:
        return "⚠️ Could not load the PDF."
    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    texts = text_splitter.split_documents(documents)
    logger.debug("Document split into %d text chunks", len(texts))

    # Create an embeddings database using Chroma from the split text chunks.
    db = Chroma.from_documents(texts, embedding=embeddings)
    logger.debug("Chroma vector store initialized.")

    # Custom prompt template to get only the answer
    prompt = PromptTemplate(
        template="""
                    You are an assistant answering questions based ONLY on the provided context.

                    Rules:
                    - Use the context silently.
                    - Do NOT repeat the context.
                    - Do NOT include CV-style listings or raw text.
                    - Answer in clear, concise sentences.
                    - If the answer is not found in the context, say:
                    "I could not find this information in the uploaded document."

                    Context:
                    {context}\n

                    Question:
                    {question}\n

                    Answer:
                    """,
        input_variables=["context", "question"]
    )

    # Optional: Log available collections if accessible (this may be internal API)
    try:
        collections = db._client.list_collections()  # _client is internal; adjust if needed
        logger.debug("Available collections in Chroma: %s", collections)
    except Exception as e:
        logger.warning("Could not retrieve collections from Chroma: %s", e)

    # Build the QA chain, which utilizes the LLM and retriever for answering questions. 
    conversation_retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm_hub,
        chain_type="stuff",
        retriever=db.as_retriever(search_type="mmr",
                                  search_kwargs={'k': 6, 'lambda_mult': 0.25}),
        chain_type_kwargs={"prompt": prompt}, # Use my prompt template
        return_source_documents=False,
        input_key="question"
        )
    
    logger.info("RetrievalQA chain created successfully.")

# Function to process a user prompt
def process_prompt(prompt):
    """
    Process a user prompt using the retrieval-based LLM chain.

    Args:
        prompt (str): The user's input/question.

    Returns:
        str: The model's response, or an informative error message if not ready.
    """
    global conversation_retrieval_chain
    global chat_history

    logger.info("Processing prompt: %s", prompt)

    # Check if the retrieval chain is initialized
    if conversation_retrieval_chain is None:
        logger.warning("Retrieval chain is not initialized. PDF may not have been processed yet.")
        return "⚠️ No document has been processed yet. Please upload a PDF first."

    # Query the model
    try:
        output = conversation_retrieval_chain.invoke({
            "question": prompt,
            "chat_history": chat_history
        })
        answer = output["result"]
    except Exception as e:
        logger.error("Error querying LLM: %s", e)
        return "⚠️ Something went wrong while processing your message."

    # Update the chat history
    chat_history.append((prompt, answer))
    logger.debug("Chat history updated. Total exchanges: %d", len(chat_history))

    return answer
# Initialize the language model
init_llm(model_id= "meta-llama/Llama-2-7b-chat-hf")

