import requests
import os
from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
    """Add a new post to the forum"""
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
    """Fetch all forum posts"""
    posts = list(db.posts.find().sort("timestamp", -1))
    for p in posts:
        p["_id"] = str(p["_id"]) 
    return {"posts": posts}


@app.post("/forum/reply")
def add_reply(post_id: str = Form(...), username: str = Form(...), reply: str = Form(...)):
    """Add a reply to a specific post"""
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
    """Fetch replies for a post"""
    replies = list(db.replies.find({"post_id": post_id}).sort("timestamp", 1))
    for r in replies:
        r["_id"] = str(r["_id"])
    return {"replies": replies}

@app.post("/chatbot")
def chatbot_response(message: str = Form(...)):
    import os, requests, json
    from fastapi.responses import JSONResponse
    from fastapi import HTTPException

    print("Received message:", message)
    api_key = os.getenv("COHERE_API_KEY")
    print(" Cohere Key Present:", bool(api_key))

    if not api_key:
        raise HTTPException(status_code=500, detail="Cohere API key missing.")

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
        print(" Sending request to Cohere /v1/chat...")
        res = requests.post("https://api.cohere.ai/v1/chat", headers=headers, json=payload)
        print(" Response status:", res.status_code)
        print(" Raw text:", res.text[:300])

        if res.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Cohere API error: {res.text}")

        data = res.json()
        reply = data.get("text") or "I'm here to listen. Can you tell me more?"
        print(" Reply:", reply.strip())

        return JSONResponse({"reply": reply.strip()})
    except Exception as e:
        print(" Exception:", e)
        raise HTTPException(status_code=500, detail=str(e))

