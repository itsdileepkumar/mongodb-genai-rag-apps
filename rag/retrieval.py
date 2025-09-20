from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pydantic import SecretStr
import key_param

# MongoDB database and collection configuration
dbName = "book_mongodb_chunks"
collectionName = "chunked_data"
index = "vector_index"   # vector search index defined in MongoDB Atlas


# -------------------------------
# 1. Initialize the Vector Store
# -------------------------------
print("[Init] Connecting to MongoDB and initializing vector store...")
vector_store = MongoDBAtlasVectorSearch.from_connection_string(
    key_param.MONGODB_URI,               # MongoDB Atlas connection string
    dbName + "." + collectionName,       # target database.collection
    AzureOpenAIEmbeddings(               # embedding model used for vector search
        disallowed_special=(),
        api_key=SecretStr(key_param.AZURE_OPENAI_API_KEY),
        azure_endpoint=key_param.AZURE_OPENAI_ENDPOINT,
        model=key_param.AZURE_OPENAI_EMBEDDING_MODEL_NAME,
        api_version=key_param.AZURE_OPENAI_API_VERSION
    ),
    index_name=index                      # which MongoDB vector index to use
)
print("[Init] Vector store initialized successfully.")


# -------------------------------
# 2. Define a query function
# -------------------------------
def query(query: str):
    """
    Takes a user query, retrieves relevant context from MongoDB Atlas,
    and generates an answer using Azure OpenAI with RAG.
    """

    print(f"\n[Query] Received user query: {query}")

    # Retriever: finds top-k similar documents in MongoDB Atlas
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 3,                                   # return top 3 results
            "pre_filter": {"hasCode": {"$eq": False}}, # filter out docs where hasCode=True
            "score_threshold": 0.01                   # ignore irrelevant docs
        }
    )
    print("[Retriever] Retriever configured (k=3, filter hasCode=False, score_threshold=0.01)")

    # Prompt template that guides the LLM behavior
    template = """
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Do not answer the question if there is no given context.
    Do not answer the question if it is not related to the context.
    Do not give recommendations to anything other than MongoDB.
    Context:
    {context}
    Question: {question}
    """
    custom_rag_propmt = PromptTemplate.from_template(template)
    print("[Prompt] Custom RAG prompt template created.")

    # Data mapping for the pipeline:
    retrieve = {
        "context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])),
        "question": RunnablePassthrough()
    } 

    # LLM setup: Azure GPT chat model
    llm = AzureChatOpenAI(
        api_key=SecretStr(key_param.AZURE_OPENAI_API_KEY),
        azure_endpoint=key_param.AZURE_OPENAI_ENDPOINT,
        api_version=key_param.AZURE_OPENAI_API_VERSION,
        model=key_param.AZURE_OPENAI_CHAT_MODEL_NAME,
        temperature=0,     # deterministic answers
        max_retries=3      # retry on failure
    )
    print("[LLM] AzureChatOpenAI initialized.")

    # Parser to extract plain text from the LLM response
    response_parse = StrOutputParser()
    print("[Parser] Output parser configured to extract plain text.")

    # -------------------------------
    # 3. Build the RAG pipeline
    # -------------------------------
    rag_chain = (
        retrieve
        | custom_rag_propmt
        | llm
        | response_parse
    )
    print("[Pipeline] RAG chain assembled (retriever → prompt → LLM → parser).")

    # -------------------------------
    # 4. Execute the chain
    # -------------------------------
    print("[Pipeline] Invoking RAG chain...")
    answer = rag_chain.invoke(query)

    return answer

print(query("When did MongoDB begin supporting multi-document transactions?"))