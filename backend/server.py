import os
import requests
from datetime import datetime
from bson import ObjectId
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from passlib.hash import bcrypt
from pymongo import MongoClient

# -------------------- APP SETUP --------------------

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


# -------------------- HELPERS --------------------

def make_chat_id(mentor_email: str, student_username: str) -> str:
    return f"{mentor_email}__{student_username}"


# -------------------- AUTH: MENTOR --------------------

@app.post("/register/mentor")
def register_mentor(email: str = Form(...), password: str = Form(...)):
    if db.users.find_one({"email": email, "type": "mentor"}):
        raise HTTPException(400, "Email already registered as a mentor.")

    db.users.insert_one({
        "email": email,
        "password": bcrypt.hash(password),
        "type": "mentor",
        "joined_on": datetime.utcnow().isoformat()
    })

    return {"message": f"Welcome, Mentor ({email})! Account created successfully."}


@app.post("/login/email")
def login_mentor(email: str = Form(...), password: str = Form(...)):
    user = db.users.find_one({"email": email})
    if not user:
        raise HTTPException(401, "Mentor not found. Please register.")

    if not bcrypt.verify(password, user["password"]):
        raise HTTPException(401, "Invalid password.")

    return {"message": f"Welcome back, {email}!"}


# -------------------- AUTH: STUDENT --------------------

@app.post("/register/anonymous")
def register_anonymous(username: str = Form(...), password: str = Form(...)):
    if db.users.find_one({"username": username}):
        raise HTTPException(400, "Nickname already taken.")

    db.users.insert_one({
        "username": username,
        "password": bcrypt.hash(password),
        "type": "anonymous",
        "joined_on": datetime.utcnow().isoformat()
    })

    return {"message": f"Welcome, {username}! Please log in to continue."}


@app.post("/login/anonymous")
def login_anonymous(username: str = Form(...), password: str = Form(...)):
    user = db.users.find_one({"username": username})
    if not user:
        raise HTTPException(401, "User not found. Please register.")

    if not bcrypt.verify(password, user["password"]):
        raise HTTPException(401, "Invalid password.")

    return {"message": f"Welcome back, {username}!", "username": username}


# -------------------- FORUM --------------------

@app.post("/forum/post")
def create_post(username: str = Form(...), message: str = Form(...)):
    user = db.users.find_one({"$or": [{"email": username}, {"username": username}]})
    user_type = user["type"] if user else "anonymous"

    db.posts.insert_one({
        "username": username,
        "message": message,
        "type": user_type,
        "timestamp": datetime.utcnow().isoformat()
    })
    return {"message": "Post added successfully!"}


@app.get("/forum/all")
def get_all_posts():
    posts = list(db.posts.find().sort("timestamp", -1))
    for p in posts:
        p["_id"] = str(p["_id"])
    return {"posts": posts}


@app.post("/forum/reply")
def add_reply(post_id: str = Form(...), username: str = Form(...), reply: str = Form(...)):
    if not db.posts.find_one({"_id": ObjectId(post_id)}):
        raise HTTPException(404, "Post not found")

    user = db.users.find_one({"$or": [{"email": username}, {"username": username}]})
    user_type = user["type"] if user else "anonymous"

    db.replies.insert_one({
        "post_id": post_id,
        "username": username,
        "reply": reply,
        "type": user_type,
        "timestamp": datetime.utcnow().isoformat()
    })

    return {"message": "Reply added"}


@app.get("/forum/replies/{post_id}")
def get_replies(post_id: str):
    replies = list(db.replies.find({"post_id": post_id}).sort("timestamp", 1))
    for r in replies:
        r["_id"] = str(r["_id"])
    return {"replies": replies}


# -------------------- CHAT SYSTEM --------------------

@app.post("/chat/start")
def start_chat(student_username: str = Form(...), mentor_email: str = Form(...)):
    return {"chat_id": make_chat_id(mentor_email, student_username)}


@app.post("/chat/send")
def send_message(chat_id: str = Form(...), sender: str = Form(...), text: str = Form(...)):

    ts = datetime.utcnow().isoformat()

    try:
        mentor_email, student_username = chat_id.split("__", 1)
    except:
        raise HTTPException(400, "Invalid chat_id format.")

    chat = db.chats.find_one({"chat_id": chat_id})
    if not chat:
        db.chats.insert_one({
            "chat_id": chat_id,
            "mentor": mentor_email,
            "student": student_username,
            "messages": [],
            "last_message": "",
            "last_timestamp": ts,
            "unread_for_student": 0,
            "unread_for_mentor": 0
        })

    msg = {"sender": sender, "text": text, "timestamp": ts}

    db.chats.update_one(
        {"chat_id": chat_id},
        {
            "$push": {"messages": msg},
            "$set": {"last_message": text, "last_timestamp": ts}
        }
    )

    if sender == mentor_email:
        db.chats.update_one({"chat_id": chat_id}, {"$inc": {"unread_for_student": 1}})
    else:
        db.chats.update_one({"chat_id": chat_id}, {"$inc": {"unread_for_mentor": 1}})

    return {"status": "Message sent"}


@app.get("/chat/{chat_id}")
def get_chat(chat_id: str):
    chat = db.chats.find_one({"chat_id": chat_id})
    if not chat:
        raise HTTPException(404, "Chat not found")

    chat["_id"] = str(chat["_id"])
    return chat


@app.get("/chat/student/{student_username}")
def get_student_chats(student_username: str):
    chats = list(db.chats.find({"student": student_username}).sort("last_timestamp", -1))

    for c in chats:
        if "chat_id" not in c:
            c["chat_id"] = make_chat_id(c["mentor"], c["student"])
            db.chats.update_one({"_id": c["_id"]}, {"$set": {"chat_id": c["chat_id"]}})
        c["_id"] = str(c["_id"])

    return {"chats": chats}


@app.get("/chat/mentor/{mentor_email}")
def get_mentor_chats(mentor_email: str):
    chats = list(db.chats.find({"mentor": mentor_email}).sort("last_timestamp", -1))

    for c in chats:
        if "chat_id" not in c:
            c["chat_id"] = make_chat_id(c["mentor"], c["student"])
            db.chats.update_one({"_id": c["_id"]}, {"$set": {"chat_id": c["chat_id"]}})
        c["_id"] = str(c["_id"])

    return {"chats": chats}


@app.post("/chat/mark_read/student/{chat_id}")
def mark_read_student(chat_id: str):
    db.chats.update_one({"chat_id": chat_id}, {"$set": {"unread_for_student": 0}})
    return {"status": "ok"}


@app.post("/chat/mark_read/mentor/{chat_id}")
def mark_read_mentor(chat_id: str):
    db.chats.update_one({"chat_id": chat_id}, {"$set": {"unread_for_mentor": 0}})
    return {"status": "ok"}


# -------------------- MENTOR PROFILE --------------------

@app.get("/mentor/profile/{email}")
def get_mentor_profile(email: str):
    mentor = db.users.find_one({"email": email, "type": "mentor"})
    if not mentor:
        raise HTTPException(status_code=404, detail="Mentor not found")

    mentor["_id"] = str(mentor["_id"])

    return {
        "email": mentor["email"],
        "name": mentor.get("name", mentor["email"].split("@")[0]),
        "occupation": mentor.get("occupation", "Mentor"),
        "age": mentor.get("age", ""),
        "bio": mentor.get("bio", "You have not added a bio yet."),
        "city": mentor.get("city", ""),
        "college": mentor.get("college", ""),
        "posts_count": db.posts.count_documents({"username": mentor["email"]}),
        "joined_on": mentor.get("joined_on", "Recently Joined")
    }


@app.post("/mentor/profile/update")
def update_mentor_profile(
    email: str = Form(...),
    name: str = Form(...),
    occupation: str = Form(...),
    age: str = Form(""),
    bio: str = Form(...),
    city: str = Form(""),
    college: str = Form("")
):
    mentor = db.users.find_one({"email": email, "type": "mentor"})
    if not mentor:
        raise HTTPException(status_code=404, detail="Mentor not found")

    update_data = {
        "name": name,
        "occupation": occupation,
        "age": age.strip(),
        "bio": bio,
        "city": city.strip(),
        "college": college.strip()
    }

    db.users.update_one(
        {"email": email, "type": "mentor"},
        {"$set": update_data}
    )

    return {"status": "Profile updated successfully!"}

# -------------------- CHATBOT --------------------

@app.post("/chatbot")
def chatbot_response(message: str = Form(...)):
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise HTTPException(500, "Cohere API key missing.")

    payload = {
        "model": "command-r-08-2024",
        "preamble":
            "You are ManasAI, an empathetic and supportive mental health "
            "companion for students. You listen kindly, validate feelings, and "
            "never give medical advice.",
        "message": message,
        "chat_history": [],
        "temperature": 0.7
    }

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        res = requests.post("https://api.cohere.ai/v1/chat", headers=headers, json=payload)
        if res.status_code != 200:
            raise Exception(res.text)

        reply = res.json().get("text", "I'm here to listen. Tell me more.")
        return {"reply": reply.strip()}

    except Exception as e:
        raise HTTPException(500, f"Cohere API error: {e}")
