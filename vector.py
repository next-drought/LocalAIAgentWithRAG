from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import json
from PyPDF2 import PdfReader
from typing import List, Dict, Optional
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDBManager:
    def __init__(self, db_root: str = "./vector_dbs"):
        self.db_root = db_root
        os.makedirs(db_root, exist_ok=True)
        self.embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    def _get_collection_path(self, topic: str) -> str:
        """Get path for topic-specific collection"""
        return os.path.join(self.db_root, topic)

    def _load_document(self, file_path: str) -> List[Document]:
        """Load a single document with automatic type detection"""
        try:
            file_path = str(Path(file_path).resolve())
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            if file_path.endswith('.pdf'):
                return self._load_pdf(file_path)
            elif file_path.endswith('.json'):
                return self._load_json(file_path)
            elif file_path.endswith('.txt'):
                return self._load_text(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_path}")
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {str(e)}")
            raise

    def _load_pdf(self, file_path: str) -> List[Document]:
        """Load text content from PDF"""
        try:
            reader = PdfReader(file_path)
            return [
                Document(
                    page_content=page.extract_text(),
                    metadata={
                        "page": i+1,
                        "source": file_path,
                        "type": "pdf"
                    },
                    id=f"pdf_{i}"
                )
                for i, page in enumerate(reader.pages)
            ]
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {str(e)}")
            raise

    def _load_json(self, file_path: str) -> List[Document]:
        """Load content from JSON file"""
        try:
            with open(file_path) as f:
                data = json.load(f)
            
            if isinstance(data, list):
                return [
                    Document(
                        page_content=str(item),
                        metadata={
                            "source": file_path,
                            "type": "json_list_item"
                        },
                        id=f"json_{i}"
                    )
                    for i, item in enumerate(data)
                ]
            elif isinstance(data, dict):
                return [Document(
                    page_content=json.dumps(data, indent=2),
                    metadata={
                        "source": file_path,
                        "type": "json_document"
                    },
                    id="json_0"
                )]
            return []
        except Exception as e:
            logger.error(f"Error loading JSON {file_path}: {str(e)}")
            raise

    def _load_text(self, file_path: str) -> List[Document]:
        """Load content from plain text file"""
        try:
            with open(file_path) as f:
                content = f.read()
            return [Document(
                page_content=content,
                metadata={
                    "source": file_path,
                    "type": "text"
                },
                id="text_0"
            )]
        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {str(e)}")
            raise

    def create_or_update_collection(
        self,
        topic: str,
        sources: List[str],
        metadata: Optional[Dict] = None
    ):
        """Create or update a topic-specific vector collection"""
        collection_path = self._get_collection_path(topic)
        is_new = not os.path.exists(collection_path)
        
        documents = []
        for source in sources:
            try:
                docs = self._load_document(source)
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} documents from {source}")
            except Exception as e:
                logger.error(f"Skipping {source}: {str(e)}")
                continue

        vector_store = Chroma(
            collection_name=topic,
            persist_directory=collection_path,
            embedding_function=self.embeddings
        )

        if is_new and documents:
            try:
                vector_store.add_documents(documents=documents)
                logger.info(f"Created new collection '{topic}' with {len(documents)} docs")
            except Exception as e:
                logger.error(f"Failed to create collection: {str(e)}")
                raise
        
        return vector_store.as_retriever(search_kwargs={"k": 5})

    def get_retriever(self, topic: str):
        """Get retriever for existing collection"""
        collection_path = self._get_collection_path(topic)
        if not os.path.exists(collection_path):
            raise ValueError(f"Collection '{topic}' does not exist")
            
        return Chroma(
            collection_name=topic,
            persist_directory=collection_path,
            embedding_function=self.embeddings
        ).as_retriever(search_kwargs={"k": 5})

# Example usage:
# manager = VectorDBManager()
# retriever = manager.create_or_update_collection(
#     topic="restaurant_reviews",
#     sources=["reviews.json", "feedback.pdf"],
#     metadata={"domain": "food", "source": "customer_feedback"}
# )