import requests
import os
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pymongo import MongoClient
from passlib.hash import bcrypt
from datetime import datetime
from bson import ObjectId

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/Frontend", StaticFiles(directory="../Frontend"), name="Frontend")
@app.get("/")
async def read_root():
    return FileResponse("../Frontend/index.html")

@app.get("/login")
async def read_login():
    return FileResponse("../Frontend/login.html")

@app.get("/dashboard")
async def read_dashboard():
    return FileResponse("../Frontend/dashboard.html")

@app.get("/mentor-dashboard")
async def read_mentor_dashboard():
    return FileResponse("../Frontend/mentor_dashboard.html")

@app.get("/forum")
async def read_forum():
    return FileResponse("../Frontend/forum.html")

@app.get("/chatbot-ui")
async def read_chatbot_ui():
    return FileResponse("../Frontend/chatbot.html")

@app.get("/stories")
async def read_stories():
    if os.path.exists("../Frontend/stories.html"):
        return FileResponse("../Frontend/stories.html")
    return JSONResponse({"detail": "Stories page coming soon!"})

@app.get("/tips")
async def read_tips():
    if os.path.exists("../Frontend/tips.html"):
        return FileResponse("../Frontend/tips.html")
    return JSONResponse({"detail": "Healthy Living tips coming soon!"})


client = MongoClient("mongodb://localhost:27017")
db = client.manas


@app.post("/register/mentor")
def register_mentor(email: str = Form(...), password: str = Form(...)):
    existing_user = db.users.find_one({"email": email, "type": "mentor"})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered as a mentor.")

    hashed_pw = bcrypt.hash(password)
    db.users.insert_one({
        "email": email,
        "password": hashed_pw,
        "type": "mentor"
    })
    return JSONResponse({"message": f"Welcome, Mentor ({email})! Account created successfully."})


@app.post("/login/email")
def login_mentor(email: str = Form(...), password: str = Form(...)):
    user = db.users.find_one({"email": email})
    if not user:
        raise HTTPException(status_code=401, detail="Mentor not found. Please register first.")

    if not bcrypt.verify(password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid password.")

    return JSONResponse({"message": f"Welcome back, {email}!"})


@app.post("/register/anonymous")
def register_anonymous(username: str = Form(...), password: str = Form(...)):
    existing = db.users.find_one({"username": username})
    if existing:
        raise HTTPException(status_code=400, detail="Nickname already taken. Try another.")

    hashed_pw = bcrypt.hash(password)
    db.users.insert_one({
        "username": username,
        "password": hashed_pw,
        "type": "anonymous"
    })
    return JSONResponse({"message": f"Welcome, {username}!, please login to continue."})


@app.post("/login/anonymous")
def login_anonymous(username: str = Form(...), password: str = Form(...)):
    user = db.users.find_one({"username": username})
    if not user:
        raise HTTPException(status_code=401, detail="User not found. Please register first.")
    if not bcrypt.verify(password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid password.")
    
    return JSONResponse({"message": f"Welcome back, {username}!",
                         "username": username})



@app.post("/forum/post")
def create_post(username: str = Form(...), message: str = Form(...)):
    user = db.users.find_one({"$or": [{"email": username}, {"username": username}]})
    user_type = user.get("type", "anonymous") if user else "anonymous"

    post = {
        "username": username,
        "message": message,
        "type": user_type,
        "timestamp": datetime.utcnow().isoformat()
    }
    db.posts.insert_one(post)
    return JSONResponse({"message": "Post added successfully!"})


@app.get("/forum/all")
def get_all_posts():
    """Fetch all forum posts, newest first"""
    posts = list(db.posts.find().sort("timestamp", -1))
    # Convert ObjectId to string for JSON compatibility
    for p in posts:
        p["_id"] = str(p["_id"]) 
    return {"posts": posts}


@app.post("/forum/reply")
def add_reply(post_id: str = Form(...), username: str = Form(...), reply: str = Form(...)):
    if not db.posts.find_one({"_id": ObjectId(post_id)}):
        return JSONResponse({"error": "Post not found"}, status_code=404)
    
    user = db.users.find_one({"$or": [{"email": username}, {"username": username}]})
    user_type = user.get("type", "anonymous") if user else "anonymous"

    reply_doc = {
        "post_id": post_id,
        "username": username,
        "reply": reply,
        "type": user_type,
        "timestamp": datetime.utcnow().isoformat()
    }
    db.replies.insert_one(reply_doc)
    return JSONResponse({"message": "Reply added successfully!"})


@app.get("/forum/replies/{post_id}")
def get_replies(post_id: str):
    replies = list(db.replies.find({"post_id": post_id}).sort("timestamp", 1))
    for r in replies:
        r["_id"] = str(r["_id"])
    return {"replies": replies}


@app.post("/chatbot")
def chatbot_response(message: str = Form(...)):
    print(f"Chatbot received: {message}")
    api_key = os.getenv("COHERE_API_KEY")
    
    if not api_key:
        print("ERROR: COHERE_API_KEY is not set.")
        raise HTTPException(status_code=500, detail="Cohere API key missing on server.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "command-r-08-2024",   
        "preamble": "You are ManasAI, an empathetic and supportive mental health companion for students. \
        You listen kindly, validate feelings, and never give medical advice.",
        "message": message,
        "chat_history": [],
        "temperature": 0.7
    }

    try:
        res = requests.post("https://api.cohere.ai/v1/chat", headers=headers, json=payload)
        
        if res.status_code != 200:
            print(f"Cohere API Error: {res.text}")
            raise HTTPException(status_code=500, detail=f"Cohere API error: {res.text}")

        data = res.json()
        reply = data.get("text") or "I'm here to listen. Can you tell me more?"
        return JSONResponse({"reply": reply.strip()})
        
    except Exception as e:
        print("Exception during chatbot request:", e)
        raise HTTPException(status_code=500, detail=str(e))