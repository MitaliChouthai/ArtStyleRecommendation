# Art Style Recommendation

![Arts](https://s.studiobinder.com/wp-content/uploads/2020/08/Types-of-Art-Styles-Featured.jpg)

##### Image Source: [StudioBinder]
----- 

[![Streamlit](https://img.shields.io/badge/Streamlit%20Application-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](http://34.125.146.230:8501/)

[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=FastAPI&logoColor=white)](http://34.125.146.230:8051/docs)

[![codelabs](https://img.shields.io/badge/codelabs-4285F4?style=for-the-badge&logo=codelabs&logoColor=white)](https://codelabs-preview.appspot.com/?file_id=1K5KXsSgMQ-jTM3fTJxYQWo1an-y66M_F1NfkNllJC_g#0)

[![Demo Link](https://img.shields.io/badge/Demo_Link-808080?style=for-the-badge&logo=YouTube&logoColor=white)](https://drive.google.com/file/d/170hl_0gA1Rog9UF96av5BJ_DmRD4bzvq/view?usp=drivesdk)

----- 

## Index
  - [Motivation 🎯](#motivation)
  - [Technical Abstract 📝](#technical-abstract)
  - [Architecture Diagram 🏗](#architecture-diagram)
  - [Project Structure 🗃️](#project-structure)
  - [Project Components 💽](#project-components)
  - [How to run the application 💻](#how-to-run-the-application-locally)
----- 

## Motivation

Art appreciation is a subjective experience, and individuals often seek personalized recommendations that align with their preferences. This project aims to develop an Arts Recommendation System that incorporates user engagement metrics, specifically time spent on each artwork, to dynamically adjust and refine recommendations. Users can interact with the chatbot to inquire about specific art-related topics, and receive information about artists or art movements.

## Technical Abstract
- Users register on the Streamlit-based Art Recommendation System, providing their preferences across three categories during the onboarding process.
- The system showcases three artworks from each of the user's selected categories and introduces an element of diversification by presenting three additional artworks from categories not chosen by the user.
- Users can click on artwork images to view detailed information, including artist name, creation year, etc., on an enlarged page.
- Leveraging OpenAI's CLIP model, the system retrieves and displays similar artworks stored in Pinecone database based on image embeddings.
- Art images are stored in Google Cloud Storage (GCS), and user-related information is managed in BigQuery, facilitating efficient storage and retrieval.
- The system calculates weighted preferences by combining initial category preferences (70%) and user interaction time (30%), dynamically updating the user's dashboard.
- Pinecone, a vector database, is employed to store and retrieve image embeddings, enabling efficient and scalable content-based recommendations using CLIP embeddings.
- A chatbot, powered by Langchain language model (LLM) and an agent with Wikipedia docstore, offers a conversational interface for art-related inquiries and information retrieval.
- The system continuously analyzes user interaction data, tracking the time spent on different categories to understand evolving preferences and adapt recommendations
- By combining recommendation diversification, detailed artwork exploration, weighted preference updates, and a conversational chatbot, the system provides users with a holistic and personalized art exploration journey.

## Architecture Diagram

![art style recommendation](https://github.com/AlgoDM-Fall2023-Team4/Final_Project/blob/pranitha_dev/architecture_diagram/arts_recommendation_system.png)

## Project Structure

```
  ├── assets           # images used for readme
  │   └── ... .png
  ├── architecture-diagram
  │   ├── architecture_diagram.py          # architectural diagram python code    
  │   └── arts_recommendation_system.png   # architectural diagram png
  ├── dashboard
  │   ├── home.py                     # code to user dashboard
  │   └── display_images.py           # code to display user selected image, similar images and chatbot
  ├── fastapi
  │   └── repository.py               # application code for Fastapi
  ├── data
  │   ├── Pinecone_Embeddings.ipynb   # notebook to generate embeddings for images and add it to Pinecone Db
  │   └── imagesinfo.csv              # data csv
  ├── main.py                         # code for streamlit application (Sign up, Sign in, Logout)
  └── requirements.txt                # libraries required to build the application
```

## Project Components

### OPENAI's CLIP:
It leverages advanced image-text embeddings to enhance art recommendations, retrieving similar artworks based on visual and textual similarities, providing a more nuanced and personalized exploration experience for users.

### Pinecone:
Pinecone serves as a vector database, efficiently storing and retrieving CLIP model embeddings, enabling fast and scalable content-based art recommendations in the recommendation system.

### Langchain:
Langchain (LLMChain) powers the conversational chatbot, providing natural language understanding and generation capabilities, facilitating interactive and informative dialogues with users.

### Wikipedia:
Docstore is integrated into the chatbot's agent, serving as a knowledge base for information retrieval, enabling the chatbot to provide contextually relevant details about artists, art movements, and historical events.

### Google Cloud Storage:
GCS is utilized for storing and managing art images, offering scalable and secure cloud storage that enables efficient retrieval and rendering of images within the Streamlit and FastAPI applications.

### Google BigQuery:
BigQuery is employed for storing and managing user-related information, supporting efficient querying and analytics to track user preferences, interaction times, and other relevant metrics for recommendation system adaptation.

### Compute Engine:
The Streamlit and FastAPI applications, serving as the frontend and backend of the art recommendation system, are hosted on Google Compute Engine, providing scalable and reliable compute resources for seamless user interaction and real-time processing.

### Streamlit
Python library Streamlit has been implemented in this application for its user interface. Streamlit offers user friendly experience to assist users in :

>  User Registration and Login

>  User dashboard

>  Display art and information related to art

>  Chatbot for users to interact and learn more

### FastAPI:
Employed FastAPI as the backend framework to ensure efficient and high-performance processing, handling user requests, computing recommendation weights, and facilitating real-time updates to the user's dashboard based on interaction metrics.

## How to run the application locally

1. Clone the repo to get all the source code on your machine

2. Change directory to the Final_Project

  - First, create a virtual environment and install all requirements from the [`requirements.txt`](https://github.com/BigDataIA-Spring2023-Team-08/assignment05-fit-finder-app/blob/main/main/requirements.txt) file present
    ```bash
      python -m venv .venv
      pip install -r requirements.txt
    ```
  - Next, get your OPENAI_API_KEY.
  - Add all necessary credentials into a `.env` file:
  ```
      OPENAI_API_KEY=XXXXX
  ```

3. Next, let us now run the application

  - First, open 2 tabs in terminal and change the directory to Final_Project and activate the virtual environment we created in the earlier step
    ```bash
        source .venv/bin/activate
    ```
  - Then, run the streamlit application locally using the main.py script:
  ```
      python -m streamlit run main.py
  ```
  - Finally, in the other terminal tab, change the directory to fastapi and run the fastapi application locally using the fastapi/repository.py:
    ```bash
        uvicorn repository:app --host 0.0.0.0 --port 8080 --reload
    ```
4. Use the Art Style Recommendation application at http://0.0.0.0:8501
