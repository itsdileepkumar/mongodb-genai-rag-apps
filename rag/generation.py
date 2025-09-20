from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.document_transformers.openai_functions import (
    create_metadata_tagger,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings, AzureChatOpenAI, AzureOpenAIEmbeddings
from pymongo import MongoClient
import rag.key_param as key_param
from pydantic import SecretStr


# Set the MongoDB URI, DB, Collection Names

client = MongoClient(key_param.MONGODB_URI)
dbName = "book_mongodb_chunks"
collectionName = "chunked_data"
collection = client[dbName][collectionName]

# Load and preprocess the PDF
loader = PyPDFLoader("./mongodb.pdf")
pages = loader.load()
cleaned_pages = []

for page in pages:
    if len(page.page_content.split(" ")) > 20:
        cleaned_pages.append(page)

print(f"Total Pages: {len(pages)}")
print(f"Cleaned Pages: {len(cleaned_pages)}")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
schema = {
    "properties": {
        "title": { "type": "string"},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "hasCode": {"type": "boolean"}
    },
    "required": ["title","keywords","hasCode"]
}

# Create the LLM and document transformer
print("Generating metadata for the documents...")
llm = AzureChatOpenAI(
    api_key=SecretStr(key_param.AZURE_OPENAI_API_KEY),
    azure_endpoint=key_param.AZURE_OPENAI_ENDPOINT,
    api_version=key_param.AZURE_OPENAI_API_VERSION,
    model=key_param.AZURE_OPENAI_CHAT_MODEL_NAME,
    temperature=0
)
document_transformer = create_metadata_tagger(schema, llm)
docs = document_transformer.transform_documents(cleaned_pages)
split_docs = text_splitter.split_documents(docs)

# Get embeddings and store in MongoDB
print("Generating embeddings and storing in MongoDB Atlas...")
embeddings = AzureOpenAIEmbeddings(
    api_key=SecretStr(key_param.AZURE_OPENAI_API_KEY),
    azure_endpoint=key_param.AZURE_OPENAI_ENDPOINT,
    model=key_param.AZURE_OPENAI_EMBEDDING_MODEL_NAME,
    api_version=key_param.AZURE_OPENAI_API_VERSION
)
vector_store = MongoDBAtlasVectorSearch.from_documents(
    split_docs,
    embeddings, collection=collection
)