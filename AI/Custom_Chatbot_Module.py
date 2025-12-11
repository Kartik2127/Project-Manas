import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

import random
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()

knowledge_base = {
    "greetings": {
        "patterns": ["hello", "hi", "hey", "I want to talk", "what's up", "good morning", "good evening"],
        "responses": [
            "Hey there, I’m really glad you reached out today. How are you feeling right now?",
            "Hi, it’s good to hear from you. I want you to know that I’m here for you, no matter what’s on your mind.",
            "Hello, friend. Sometimes just starting a conversation can be the hardest part, but you’ve already done that. I’m here and ready to listen."
        ],
    },

    "anxiety": {
        "patterns": ["anxiety", "anxious", "panic", "worry", "stressed", "nervous", "overthinking"],
        "responses": [
            "I can imagine how overwhelming it must feel when anxiety takes over. You don’t have to go through this alone—I’m right here with you. Let’s try something gentle together: take a slow, deep breath in… hold it for a moment… and then breathe out slowly, letting a bit of that tension go.",
            "It’s okay to feel anxious—those feelings don’t make you weak, they make you human. Sometimes, just naming what we feel takes away some of its power. If you’d like, I can guide you through a simple calming exercise to help bring you back to the present.",
            "I know anxiety can feel like a storm in your chest or mind. Please remember: you’re safe in this moment. You don’t have to figure it all out right now. Let’s just focus on one small breath at a time, together."
        ],
    },

    "loneliness": {
        "patterns": ["lonely", "alone", "isolated", "no one to talk to", "abandoned"],
        "responses": [
            "Feeling lonely can be really heavy, like carrying a quiet weight that no one else sees. I want you to know that right now, in this moment, you are not alone—I’m here with you, listening and caring about what you’re going through.",
            "Sometimes loneliness can make it feel like the whole world is far away. But reaching out, even just to me, shows a lot of strength. I’d love to hear more about what’s on your heart—you don’t have to hold it in by yourself.",
            "I hear you. Feeling alone is deeply painful, but your feelings matter to me. If you’d like, we can talk through what’s making you feel isolated, or we can just sit together in this space until it feels a little lighter."
        ],
    },

    "depression": {
        "patterns": ["depressed", "sad", "empty", "hopeless", "worthless", "unmotivated", "can't go on"],
        "responses": [
            "I’m so sorry you’re going through this. Depression can make even the simplest things feel impossible, and I want you to know that I see how hard you’re trying just by talking to me. You are not a burden—you deserve kindness and care, especially from yourself.",
            "When everything feels heavy, it can seem like there’s no way forward. But even the smallest step—like sharing how you feel with me right now—is a sign of hope. You matter more than you know, and I’m here to remind you of that when it’s hard to believe.",
            "It sounds like you’re in a lot of pain, and that must feel exhausting. Please remember: your feelings are valid, but they don’t define your worth. You are important, and the world is better with you in it."
        ],
    },

    "self_esteem": {
        "patterns": ["I hate myself", "I'm not good enough", "failure", "useless", "unworthy"],
        "responses": [
            "It hurts my heart to hear that you feel this way about yourself. I want you to know something: you are so much more than the negative thoughts in your head. You bring unique value to this world, even on days when you can’t see it yourself.",
            "We all struggle and make mistakes, but that never takes away from our worth as human beings. You are enough, exactly as you are, and it’s okay if you’re still learning to believe that. Until then, I can believe it for you.",
            "I care about you, and I want you to be gentle with yourself. Imagine how you’d comfort a friend who felt this way—because you deserve that same compassion too."
        ],
    },

    "motivation": {
        "patterns": ["unmotivated", "tired of life", "can't focus", "I give up", "no energy"],
        "responses": [
            "I know it’s really hard when motivation just isn’t there. Please don’t be too hard on yourself—rest is also a form of progress. Sometimes even the tiniest step, like drinking a glass of water or standing up for a stretch, can make things feel a little lighter.",
            "It’s okay if you don’t have it all figured out right now. You don’t have to climb the whole mountain today—just focus on the next small step in front of you. And I’ll be right here to cheer you on.",
            "You’re doing your best, even if it doesn’t feel like it. Taking things one moment at a time is still moving forward, and that’s something to be proud of."
        ],
    },

    "anger": {
        "patterns": ["angry", "furious", "rage", "mad", "annoyed", "frustrated"],
        "responses": [
            "Anger is such a powerful emotion—it usually means something important inside you is hurting. It’s okay to feel angry, and you’re not wrong for having that feeling. Do you want me to share a calming technique with you?",
            "I hear the frustration in your words, and I want you to know that it’s safe to let those feelings out here. Sometimes writing down what you’d like to say, or even just taking a few slow breaths, can help release some of the tension.",
            "Your anger doesn’t make you a bad person—it just means something matters to you deeply. I’m here to listen if you want to talk about it."
        ],
    },

    "sleep": {
        "patterns": ["can't sleep", "insomnia", "bad dreams", "restless", "nightmare"],
        "responses": [
            "I’m sorry you’re having trouble sleeping—it can feel so draining when your mind won’t quiet down. Maybe try imagining a calm, safe place as you breathe slowly. Sometimes that can help ease the mind into rest.",
            "Rest is so important, and it’s frustrating when it doesn’t come easily. If you’d like, I can guide you through a short relaxation exercise to help your body unwind.",
            "Even if sleep is difficult right now, know that simply lying down and resting is still giving your body some care. You deserve rest, and I hope peace comes to you soon."
        ],
    },

    "grounding_exercise": {
        "patterns": ["grounding", "calm down", "panic attack", "help me focus"],
        "responses": [
            "Let’s try a grounding exercise together. Look around you and name 5 things you can see, 4 things you can touch, 3 things you can hear, 2 things you can smell, and 1 thing you can taste. This can help bring your mind gently back to the present moment.",
            "When panic feels overwhelming, grounding yourself can help. Try placing your feet flat on the floor and notice the support beneath you. Focus on your breath, slow and steady, and remind yourself: ‘I am safe right now.’"
        ],
    },

    "grief": {
        "patterns": ["i miss them", "i lost someone", "i can’t move on", "death", "passed away"],
        "responses": [
            "Losing someone you love is incredibly hard. Would you like to share a memory about them?",
            "Grief takes time. It’s okay to feel this pain, and it shows how much you cared."
        ]
    },

    "affirmations": {
        "patterns": ["positive thoughts", "say something good", "encourage me", "uplift me"],
        "responses": [
            "Here’s something I truly believe: you are worthy of love, peace, and kindness—especially from yourself. Even on your hardest days, you are still enough.",
            "You’ve made it through challenges before, and that shows how strong you are, even if you don’t feel strong right now. I believe in you.",
            "You bring something unique and irreplaceable to this world. Please don’t forget that your presence matters."
        ],
    },

    "confusion": {
        "patterns": ["i'm lost", "i don’t know what to do", "i feel stuck", "i'm confused"],
        "responses": [
            "It sounds like you’re facing uncertainty. Do you want to talk through your options?",
            "Sometimes confusion is the first step to clarity. What’s on your mind?"
        ]
    },

    "crisis": {
        "patterns": ["suicide", "end my life", "kill myself", "want to die", "can't go on", "hopeless", "self-harm", "hurt myself", "pain is too much", "goodbye"],
        "responses": [],  # handled separately
    },

    "gratitude": {
        "patterns": ["thank you", "thanks", "i'm grateful", "appreciate it"],
        "responses": [
            "You’re most welcome. Gratitude is powerful—keep nurturing it!",
            "I’m glad I could be here for you. What else are you grateful for today?"
        ]
    },

    "default": {
        "patterns": [],
        "responses": [
            "Thank you for sharing that with me. I might not have the perfect answer, but I care deeply about what you’re going through and I’m here to listen.",
            "I may not fully understand, but I want to. Can you tell me a little more about what’s been on your mind?",
            "What you’re feeling sounds important, and I’d really like to sit with you in it for a while. You don’t have to go through this alone."
        ],
    }
}

conversation_state = {"last_intent": None, "awaiting_followup": False}

def get_response(user_message, knowledge_base, threshold=1):
    global conversation_state
    user_message = user_message.lower()
    tokens = word_tokenize(user_message)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    if conversation_state["awaiting_followup"]:
        conversation_state["awaiting_followup"] = False
        return f"Thank you for sharing: {user_message}. That means a lot."

    best_intent = None
    best_score = 0

 
    for intent, data in knowledge_base.items():
        patterns = data["patterns"]
        score = 0
        for pattern in patterns:
            pattern = pattern.lower()
            pattern_tokens = [lemmatizer.lemmatize(word) for word in word_tokenize(pattern)]
            match_count = sum(1 for word in pattern_tokens if word in lemmatized_tokens)
            score += match_count
        if score > best_score:
            best_score = score
            best_intent = intent

    if best_intent and best_score >= threshold:
        responses = knowledge_base[best_intent]["responses"]
        if responses:
            response = random.choice(responses)

            if best_intent in ["grief", "gratitude", "joy"]:
                conversation_state["awaiting_followup"] = True

            conversation_state["last_intent"] = best_intent
            return response

        elif best_intent == "crisis":
            return ("It sounds like you're going through a lot right now. "
                    "Please connect with a professional who can help. "
                    "You can call the 24/7 Suicide Prevention and Mental Health Helpline.")

    return random.choice(knowledge_base["default"]["responses"])


if __name__ == "__main__":
    print("Bot: Hi, I'm here to listen. You can type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("Bot: Take care!")
            break
        response = get_response(user_input, knowledge_base)
        print("Bot:", response)
