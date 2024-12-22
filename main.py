from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from sqlalchemy.orm import Session
import httpx
import os
from dotenv import load_dotenv
from models import User, Portfolio, Orders, get_db
from pydantic import BaseModel

app = FastAPI()

# Load environment variables
load_dotenv()
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI")
FRONTEND_BASE_URL = os.getenv("FRONTEND_BASE_URL")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://mctl.me"],  # Update as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root(request: Request, db: Session = Depends(get_db)):
    user_email = request.cookies.get("user_email")
    if not user_email or not db.query(User).filter(User.email == user_email).first():
        return RedirectResponse(f"{FRONTEND_BASE_URL}/login")
    return RedirectResponse(f"{FRONTEND_BASE_URL}/chat")


@app.get("/auth/google")
async def google_login():
    google_oauth_url = (
        f"https://accounts.google.com/o/oauth2/auth"
        f"?client_id={GOOGLE_CLIENT_ID}"
        f"&redirect_uri={REDIRECT_URI}"
        f"&response_type=code"
        f"&scope=openid email profile"
    )
    return JSONResponse(content={"redirect_url": google_oauth_url})


@app.get("/auth/google/callback")
async def google_callback(request: Request, db: Session = Depends(get_db)):
    code = request.query_params.get("code")
    if not code:
        raise HTTPException(status_code=400, detail="Code not found")

    # Exchange code for tokens
    token_url = "https://oauth2.googleapis.com/token"
    data = {
        "code": code,
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "redirect_uri": REDIRECT_URI,
        "grant_type": "authorization_code",
    }

    async with httpx.AsyncClient() as client:
        token_response = await client.post(token_url, data=data)
        token_response.raise_for_status()
        tokens = token_response.json()

    # Fetch user info
    async with httpx.AsyncClient() as client:
        user_info = await client.get(
            "https://www.googleapis.com/oauth2/v1/userinfo",
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )
        user_info.raise_for_status()
        user = user_info.json()

    # Check or create user in the database
    existing_user = db.query(User).filter(User.email == user["email"]).first()
    if not existing_user:
        new_user = User(email=user["email"], name=user.get("name", "Unknown"))
        db.add(new_user)
        db.commit()

    # Respond with JSON data
    return JSONResponse(
        content={
            "user_email": user["email"],
            "redirect_url": f"{FRONTEND_BASE_URL}/chat",
        }
    )


@app.post("/chat")
async def chat_prompt(request: Request):
    data = await request.json()
    prompt = data.get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    # Simulated AI response
    response = f"AI Response to: {prompt}"
    return {"response": response}


@app.post("/order")
async def place_order(request: Request, db: Session = Depends(get_db)):
    data = await request.json()
    user_email = request.cookies.get("user_email")

    user = db.query(User).filter(User.email == user_email).first()
    if not user:
        return {"status": "error", "message": "User not found"}

    symbol = data.get("symbol")
    quantity = data.get("quantity", 0)

    if not symbol or quantity <= 0:
        raise HTTPException(status_code=400, detail="Invalid order data")

    # Create new order
    new_order = Orders(symbol=symbol, quantity=quantity, user_id=user.id)
    db.add(new_order)

    # Update portfolio
    portfolio = (
        db.query(Portfolio)
        .filter(Portfolio.user_id == user.id, Portfolio.symbol == symbol)
        .first()
    )
    if not portfolio:
        portfolio = Portfolio(symbol=symbol, quantity=0, user_id=user.id)
        db.add(portfolio)
    portfolio.quantity += quantity
    db.commit()

    return {"status": "ok"}


@app.get("/auth/status")
async def auth_status(request: Request):
    user_email = request.cookies.get("user_email")
    return {"authenticated": user_email is not None}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
