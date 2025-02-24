from diagrams.gcp.analytics import BigQuery
from diagrams.programming.framework import Fastapi as Fastapi
from diagrams import Diagram, Cluster, Edge
from diagrams.gcp.storage import Storage
from diagrams.onprem.container import Docker
from diagrams.onprem.client import User
from diagrams.custom import Custom

with Diagram("Arts Recommendation System", show=False, direction="LR"):

    with Cluster("Google Cloud Platform"):
        with Cluster("GCP Compute Engine"):
            frontend = Custom("Streamlit (Frontend)", "assets/streamlit.png")
            backend = Fastapi("Fastapi (Backend)")
            frontend - Edge(label = "API calls", color = "red", style = "dashed") - backend

        with Cluster("Google Cloud Storage"):
            gcs = Storage("Google Cloud Storage")

        with Cluster("Databases"):
            user_tables = BigQuery("User Tables")

    with Cluster("Machine Learning"):
        clip_model = Custom("OpenAI CLIP Model", "assets/chatgpt.png")
        pinecone = Custom("Pinecone Vector DB", "assets/pinecone.png")



    with Cluster("Chatbot"):
        langchain_llm = Custom("Langchain LLM", "assets/langchain.png")
        wikipedia_agent = Custom("Wikipedia Docstore Agent", "assets/wikipedia.png")

    user = User("User")

    user << Edge(label = "User log in & getting recommendations") >> frontend

    frontend >> Edge(label = "Similar Images") >> clip_model
    backend >> Edge(label = "Fetching user related information") >> user_tables
    backend >> Edge(label = "Fetching Imags") >> gcs
    pinecone << Edge(label = "Embeddings & Similarity score") << clip_model
    frontend >> Edge(label = "Conversational Retrieval for Q&A") >> langchain_llm
    langchain_llm >> wikipedia_agent
