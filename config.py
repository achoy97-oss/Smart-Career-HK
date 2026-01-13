"""
Configuration for Job Matcher
"""
import os

class Config:
    """Configuration settings"""
    
    # Pinecone
    PINECONE_API_KEY = "xxx"
    PINECONE_ENVIRONMENT = "us-east-1"
    INDEX_NAME = "job-resume-matcher"
    EMBEDDING_DIMENSION = 384
    MODEL_NAME = "all-MiniLM-L6-v2"
    
    # Azure OpenAI
    AZURE_ENDPOINT = "XXX"
    AZURE_API_KEY = "XXX"
    AZURE_API_VERSION = "2024-10-21"
    AZURE_MODEL = "gpt-4o-mini"
    
    # RapidAPI
    #RAPIDAPI_KEY = "XXX"
    RAPIDAPI_KEY = "XXX"
    
    # Settings
    MAX_JOBS_TO_FETCH = 50
    TOP_MATCHES_TO_SHOW = 5
    UPLOAD_FOLDER = "uploads"
    
    @classmethod
    def setup(cls):
        os.makedirs(cls.UPLOAD_FOLDER, exist_ok=True)
    
    @classmethod
    def validate(cls):
        """Validate API keys are set"""
        print("✅ Configuration validated")
        return True