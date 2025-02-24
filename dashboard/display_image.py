import streamlit as st
import time
import requests
import pandas as pd
import os
from PIL import Image
import torch
from io import BytesIO
from google.cloud import storage
import clip
import time
from langchain import LLMChain, OpenAI
from langchain import Wikipedia
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.agents.react.base import DocstoreExplorer
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

PREFIX = "http://0.0.0.0:8080"

def display_images(token: str):
    user_data = {
        "token": token
    }
    response = requests.post(f"{PREFIX}/user", json=user_data)
    user = response.json()

    if "start_time" not in st.session_state:
        st.session_state['start_time'] = None

    if not st.session_state.start_time:
        start_time = start_timer()
        st.session_state.start_time = start_time

    df = pd.read_csv('./data/imagesinfo.csv')

    df['filename_numeric'] = df['filename'].str.replace('.jpg', '').astype(int)
    df = df.sort_values(by="filename_numeric")

    df = df.head(5001)

    df_cleaned = df.dropna(subset=["genre"])
    # st.write(df_cleaned)
    # st.write(st.session_state.selected_image)
    s = st.session_state.selected_image
    # st.write(s)
    filename = os.path.basename(s)
    image = load_image_from_gcs('finalproject_images', s)
    # st.write(filename)

    st.image(image, width=800)
    st.subheader(":blue[Artist Name :] " + str(df_cleaned[df_cleaned["filename"] == filename]['artist'].iloc[0]))
    st.subheader(":blue[Title :] " + str( df_cleaned[df_cleaned["filename"] == filename]['title'].iloc[0]))
    st.subheader(":blue[Genre :] " + str( df_cleaned[df_cleaned["filename"] == filename]['genre'].iloc[0]))
    st.subheader(":blue[Date :] " + str( df_cleaned[df_cleaned["filename"] == filename]['date'].iloc[0]))
    st.subheader(":blue[Style :] " + str( df_cleaned[df_cleaned["filename"] == filename]['style'].iloc[0]))

    with st.expander('Similar Images:'):
        image_embeddings = compute_clip_features(image)
        image_embeddings_list = image_embeddings.tolist()
        try:
            response = requests.post( f'{PREFIX}/get_closest_images/', json={"embeddings": image_embeddings_list, "num": 5})
            similar_images = []
            for i in response.json():
                # current_path = '/Users/pranitha/Desktop/AlgoDM/FinalProject/images/data/' + i + '.jpg'
                image = load_image_from_gcs('finalproject_images', i + '.jpg')
                similar_images.append(image)
            k=1
            for col in st.columns(4):
                col.image(similar_images[k])
                k += 1
            # display_image_from_local(response.json())
        except requests.HTTPError as e:
            st.write(e.response.status_code)
            st.write(e.response.json())

    home_button = st.button('Go to Home Page')

    if home_button:
        time_spent = stop_timer(st.session_state.start_time)
        time_spent_minutes = round(time_spent / 60, 1)
        # st.write(time_spent_minutes)
        category_dict = {
            "email": user['email'],
            "category": df_cleaned[df_cleaned["filename"] == filename]['genre'].iloc[0],
            "timespent": time_spent_minutes,
            "weight": 0.0
        }
        res = requests.post(f"{PREFIX}/timespent_update", json=category_dict)
        if res.status_code == 200:
            time.sleep(2)
            st.session_state.go_to_display_image = False
            st.session_state.selected_image = None
            st.session_state.start_time = None
            st.session_state.main_dashboard_dict = None
            st.session_state.more_dashboard_dict = None
            st.experimental_rerun()

    docstore = DocstoreExplorer(Wikipedia())
    tools = [
        Tool(
            name="Search",
            func=docstore.search,
            description="useful for when you need to ask with search",
        )
    ]
    prefix = """Have a conversation with human, answering the following questions as best as you can. You have access to the following tools:"""
    suffix = """Begin!

        {chat_history}
        Question: {input}
        {agent_scratchpad}"""

    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "chat_history", "agent_scratchpad"]
    )

    llm_chain = LLMChain(llm=OpenAI(temperature=0, model_name='gpt-3.5-turbo'), prompt=prompt)

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history")

    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)

    agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True,
                                                     memory=st.session_state.memory, handle_parsing_errors=True)

    st.title('Learn More')
    if st.button('Clear Conversation'):
        st.session_state.messages = []
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask anything"):

        st.session_state.messages.append({"role": "user", "content": prompt})
        # st.write(st.session_state.messages)

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            with st.spinner('Wait for it...'):
                assistant_response = agent_chain.run(prompt)

            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})



def display_image_from_local(text_path):
    for i,j in enumerate(text_path):
        try:
            current_path = '/Users/pranitha/Desktop/AlgoDM/FinalProject/images/data/' + j + '.jpg'
            image1 = Image.open(current_path)
            st.image(image1, width = 100)
        except Exception as e:
            st.write("Ecveption" + str(e))

def compute_clip_features(image):
    images_preprocessed = torch.stack((preprocess(image),)).to(device)

    with torch.no_grad():
        images_features = model.encode_image(images_preprocessed)
        images_features /= images_features.norm(dim=-1, keepdim=True)

    return images_features

def load_image_from_gcs(bucket_name, object_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(object_name)

    image_bytes = blob.download_as_bytes()

    image = Image.open(BytesIO(image_bytes))

    return image

def start_timer():
    return time.time()

def stop_timer(start_time):
    end_time = time.time()
    time_spent = end_time - start_time
    return time_spent
