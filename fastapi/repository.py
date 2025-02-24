from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import List
from google.cloud import bigquery
import hashlib
from fastapi.responses import JSONResponse
import numpy as np
import torch
import clip
import pinecone

pinecone.init(api_key="47c398f4-f60c-4427-adc8-d4d18359f5e9", environment="gcp-starter")
index_name = "wikiart"
index = pinecone.Index(index_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)


def encode_search_query(search_query):
    with torch.no_grad():
        text_encoded = model.encode_text(clip.tokenize(search_query).to(device))
        text_encoded /= text_encoded.norm(dim=-1, keepdim=True)
    return text_encoded

def search_closest_image(text_encoded, num):
    return index.query(
        vector=text_encoded.tolist(),
        top_k=num,
        include_values=True
    )

app = FastAPI()

project_id = "charged-formula-405300"
dataset_id = "finalproject"
table_id = "users"

table_timespent = "timespent"

client = bigquery.Client(project=project_id)


class UserCreate(BaseModel):
    name: str
    email: str
    password: str
    preferences: List[str]


class User(BaseModel):
    name: str
    email: str
    preferences: List[str]


class UserLogin(BaseModel):
    email: str
    password: str


class TokenRequest(BaseModel):
    token: str

class TimeSpentCategory(BaseModel):
    email: str
    category: str
    timespent: float
    weight: float

n_results_per_query = 1

class EmbeddingsRequest(BaseModel):
    embeddings: List[List[float]]
    num: int

@app.get("/get_closest_image/", response_class=JSONResponse)
async def get_closest_image(text: str):
    text_encoded = encode_search_query(text)
    return [search_closest_image(text_encoded, 1)['matches'][i]['id'] for i in range(1)]

@app.post("/get_closest_images/")
async def get_closest_images(data: EmbeddingsRequest):
    input_np_arr = np.array(data.embeddings)
    return [search_closest_image(input_np_arr, data.num)['matches'][i]['id'] for i in range(data.num)]

def create_user(user: UserCreate):
    hashed_password = hashlib.sha256(user.password.encode()).hexdigest()

    # Insert user data into BigQuery
    query = f"""
    INSERT INTO `{project_id}.{dataset_id}.{table_id}` (name, email, password_hash, preferences)
    VALUES ('{user.name}', '{user.email}', '{hashed_password}', {user.preferences})
    """
    query_job = client.query(query)

    # Return the created user
    return User(name=user.name, email=user.email, preferences=user.preferences)


def get_user(email: str):
    # Retrieve user data from BigQuery
    # print(email)
    query = f"""
    SELECT name, email, password_hash, preferences FROM `{project_id}.{dataset_id}.{table_id}`
    WHERE email = '{email}'
    """
    # print(query)
    query_job = client.query(query)

    # Check if the user exists
    user_rows = query_job.result()
    user_data = list(user_rows)
    # print(user_data)
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")
    return UserCreate(name=user_data[0]["name"], email=user_data[0]["email"], preferences=user_data[0]["preferences"],
                      password=user_data[0]["password_hash"])


def get_user_from_token(token: str):
    # Retrieve user data from BigQuery
    query = f"""
    SELECT name, email, preferences, password_hash FROM `{project_id}.{dataset_id}.{table_id}`
    WHERE password_hash = '{token}'
    """
    query_job = client.query(query)

    # Check if the user exists
    user_rows = query_job.result()
    user_data = list(user_rows)
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")

    return UserCreate(name=user_data[0]["name"], email=user_data[0]["email"], preferences=user_data[0]["preferences"],
                      password=user_data[0]['password_hash'])


@app.post("/signup", response_model=User)
def sign_up(user: UserCreate):
    return create_user(user)


@app.post("/login", response_model=str)
def login(user: UserLogin):
    # Retrieve user data from BigQuery
    user_details = get_user(user.email)
    # print(user_details)

    # Validate the password
    hashed_password = hashlib.sha256(user.password.encode()).hexdigest()
    # print(hashed_password)
    if hashed_password == user_details.password:
        # Return a simple token (hashed password) for demonstration purposes
        return hashed_password

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect email or password",
        headers={"WWW-Authenticate": "Bearer"},
    )


@app.post("/user", response_model=UserCreate)
def user(tokenData: TokenRequest):
    user = get_user_from_token(tokenData.token)
    print(user)
    return user

@app.get("/top_categories/{email}", response_model=List[str])
def top_categories(email: str):
    query = f"SELECT category from `{project_id}.{dataset_id}.{table_timespent}` where email = '{email}' and version in (select max(version) from `{project_id}.{dataset_id}.{table_timespent}` where email = '{email}') order by weight desc limit 3"
    query_job = client.query(query)
    rows = query_job.result()
    categories = []
    for row in rows:
        categories.append(row[0])
    print(categories)
    return categories

# max_version_query = f"SELECT version from {project_id}.{dataset_id}.{table_timespent} where email = '{categories[0].email}"
# query_job = client.query(query)
# rows = query_job.result()
# print(row[0])

@app.post("/timespent")
def post_timespent(categories: List[TimeSpentCategory]):

    rows_to_insert = []

    for data in categories:
        row = {
            "email": data.email,
            "category": data.category,
            "timespent": data.timespent,
            "weight": data.weight,
            "version": 0
        }
        rows_to_insert.append(row)

    table_id = f"{project_id}.{dataset_id}.{table_timespent}"
    errors = client.insert_rows_json(table_id, rows_to_insert)

    if errors:
        print(f"Error inserting rows: {errors}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    else:
        print("Rows inserted successfully")
        return {"message": "User categories registered successfully"}


@app.post("/timespent_update")
def update_timespent(category: TimeSpentCategory):
    query = f"SELECT * from {project_id}.{dataset_id}.{table_timespent} where email = '{category.email}' and version in (select max(version) from {project_id}.{dataset_id}.{table_timespent} where email = '{category.email}')"
    query_job = client.query(query)
    rows = query_job.result()
    ll = []

    for row in rows:
        if row[1] == category.category:
            dd = {
                "email": row[0],
                "category": row[1],
                "timespent": category.timespent,
                "weight": row[3] + (category.timespent) * 0.3,
                "version": row[4] + 1
            }
        else:
            dd = {
                "email": row[0],
                "category": row[1],
                "timespent": row[2],
                "weight": row[3],
                "version": row[4] + 1
            }
        ll.append(dd)

    print(len(ll))
    table_id = f"{project_id}.{dataset_id}.{table_timespent}"
    errors = client.insert_rows_json(table_id, ll)

    if errors:
        print(f"Error inserting rows: {errors}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    else:
        print("Rows inserted successfully")
        return {"message": "User categories updated with new version successfully"}
    # max_version_query = f"SELECT version from {project_id}.{dataset_id}.{table_timespent} where email = '{categories[0].email}"
    # query_job = client.query(max_version_query)
    # rows = query_job.result()
    # print(rows[0])
    # return rows[0]
