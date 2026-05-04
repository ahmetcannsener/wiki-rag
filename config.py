# Models
LLM_MODEL = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"

# Chunking
CHUNK_SIZE = 500        # approximate word count
CHUNK_OVERLAP = 50      # approximate word overlap

# Retrieval
TOP_K = 5           # chunks to retrieve when a specific entity is named
BROAD_TOP_K = 8     # chunks for exploratory queries with no specific entity

# Storage
SQLITE_DB_PATH = "./wikipedia.db"
CHROMA_DB_PATH = "./chroma_db"
PEOPLE_COLLECTION = "people_store"
PLACES_COLLECTION = "places_store"

# Entities to ingest
PEOPLE = [
    "Albert Einstein",
    "Marie Curie",
    "Leonardo da Vinci",
    "William Shakespeare",
    "Ada Lovelace",
    "Nikola Tesla",
    "Lionel Messi",
    "Cristiano Ronaldo",
    "Taylor Swift",
    "Frida Kahlo",
    "Isaac Newton",
    "Charles Darwin",
    "Cleopatra",
    "Napoleon Bonaparte",
    "Mahatma Gandhi",
    "Nelson Mandela",
    "Aristotle",
    "Vincent van Gogh",
    "Wolfgang Amadeus Mozart",
    "Stephen Hawking",
]

PLACES = [
    "Eiffel Tower",
    "Great Wall of China",
    "Taj Mahal",
    "Grand Canyon",
    "Machu Picchu",
    "Colosseum",
    "Hagia Sophia",
    "Statue of Liberty",
    "Pyramids of Giza",
    "Mount Everest",
    "Stonehenge",
    "Angkor Wat",
    "Petra",
    "Acropolis of Athens",
    "Chichen Itza",
    "Niagara Falls",
    "Victoria Falls",
    "Amazon rainforest",
    "Sahara",
    "Venice",
]
