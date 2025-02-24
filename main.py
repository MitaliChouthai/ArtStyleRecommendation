import pandas as pd
import requests
import streamlit as st

from dashboard.home import home
from dashboard.display_image import display_images

# TODO: 1. search bar
# TODO: 2. similar items using CLIP - done
# TODO: 3: Chatbot - done

df = pd.read_csv('data/imagesinfo.csv')

df['filename_numeric'] = df['filename'].str.replace('.jpg', '').astype(int)
df = df.sort_values(by = "filename_numeric")

df = df.head(5001)

df_cleaned = df.dropna(subset = ["genre"])

genre_list = df_cleaned["genre"].drop_duplicates().tolist()

PREFIX = "http://0.0.0.0:8080"


def fill_timespent_table(genre_list, selected_genres, email):
    category_dicts_list = []
    for i in range(len(genre_list)):
        category_dict = {
            "email": email,
            "category": genre_list[i],
            "timespent": 0.0,
            "weight": 0.0
        }
        if genre_list[i] in selected_genres:
            category_dict['weight'] = 0.7
        else:
            category_dict['weight'] = 0.0
        category_dicts_list.append(category_dict)
    response = requests.post(f"{PREFIX}/timespent", json=category_dicts_list)
    return response



def signup():
    st.title("Sign Up")
    name = st.text_input("Enter Name")
    email = st.text_input("Enter Email")
    password = st.text_input("Enter password", type="password")
    selected_genres = st.multiselect("Select preferred art genres:", genre_list, max_selections=3)
    if name and email and password and selected_genres and st.button("Sign up"):
        user = {"name": name, "password": password, "email": email, "preferences": selected_genres}
        response = requests.post(f"{PREFIX}/signup", json=user)

        if response.status_code == 200:
            user = response.json()
            st.success("You have successfully signed up!")
            st.write("Your name is:", user["name"])
            res = fill_timespent_table(genre_list, selected_genres, email)
            st.write("Your initial preferences have been saved")
        elif response.status_code == 400:
            st.error(response.json()["detail"])
        else:
            st.error("Something went wrong")


def signin():
    st.title("Sign In")
    email = st.text_input("Enter email")
    password = st.text_input("Enter password", type="password")

    if email and password and st.button("Sign in"):
        user = {
            "email": email,
            "password": password
        }
        response = requests.post(
            f"{PREFIX}/login",
            json=user
        )
        print(response)
        if response.status_code == 200:
            # access_token = response.json()["access_token"]
            access_token = response.json()
            st.success("You have successfully signed in!")
            return access_token
        elif response.status_code == 400:
            st.error(response.json()["detail"])
        else:
            st.error("Something went wrong")

pages = {
    "HOME": home,
    "DISPLAY IMAGE": display_images
}

def main():
    if "go_to_display_image" not in st.session_state:
        st.session_state.go_to_display_image = False

    st.set_page_config(
        page_title="Art Style Recommendation", page_icon=":art:", layout="wide"
    )
    st.sidebar.title("Navigation")

    # Check if user is signed in
    token = st.session_state.get("token", None)
    print('Token ::::', token)
    # Render the navigation sidebar
    if token is not None:
        selection = st.sidebar.radio("Go to", ["HOME", "Log Out"])
    else:
        selection = st.sidebar.radio("Go to", ["Sign In", "Sign Up"])

    # Render the selected page or perform logout
    if selection == "Log Out":
        st.session_state.clear()
        st.sidebar.success("You have successfully logged out!")
        st.experimental_rerun()
    elif selection == "Sign In":
        token = signin()
        if token is not None:
            st.session_state.token = token
            print(token)
            st.experimental_rerun()
    elif selection == "Sign Up":
        signup()
    else:
        if st.session_state.go_to_display_image:
            page = pages["DISPLAY IMAGE"]
            page(token)
        else:
            page = pages[selection]
            page(token)


if __name__ == "__main__":
    main()
