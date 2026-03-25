import os
from pymongo import MongoClient
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

# MongoDB Configuration
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGODB_DB_NAME", "bpt_chatbot")

mongo_client = MongoClient(MONGO_URI)
db = mongo_client[MONGO_DB]
docs_collection = db["documents"]
chats_collection = db["chats"]

# Qdrant Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)
