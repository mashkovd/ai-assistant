from __future__ import annotations as _annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, TypeVar

import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from jose import jwt
from pydantic import BaseModel, Field, TypeAdapter
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import (ModelMessage, ModelRequest, ModelResponse,
                                  TextPart, UserPromptPart)
from sqlalchemy.orm import Session
from typing_extensions import ParamSpec, TypedDict

from models import Orders, Portfolio, User, get_db

THIS_DIR = Path(__file__).parent
FRONTEND_BASE_URL = os.getenv("FRONTEND_BASE_URL")
BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL")


class Action(str, Enum):
    BUY = "buy"
    SELL = "sell"


class SupportRequest(BaseModel):
    ticker: str = Field(description="The ticker from request")
    amount: int = Field(description="The amount from request")
    rate: float = Field(description="The rate of the ticker from get_tickers tools")
    alias: Optional[str] = Field(
        description="The alias of the ticker from get_tickers tools"
    )
    action: Action


agent = Agent(
    "openai:gpt-4o",
    system_prompt="As agent just extract real ticker from stocks and amount from the request",
    result_type=SupportRequest,
)


@dataclass
class Dependencies:
    http_client: httpx.AsyncClient


@agent.tool
async def get_tickers(ctx: RunContext[Dependencies]) -> list[dict[str, Any]]:
    """Get the list of tickers."""
    response = await ctx.deps.http_client.get(
        "https://app.libertex.com/spa/instruments",
    )
    response.raise_for_status()
    data = response.json()
    result = [
        {"alias": d["alias"], "symbol": d["symbol"], "rate": d["rate"]}
        for d in data["instruments"]
    ]
    return result


SECRET_KEY = "abracadabra"
ALGORITHM = "HS256"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://192.168.1.6:5173",
        "https://mctl.me",
    ],  # Replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MessageTypeAdapter: TypeAdapter[ModelMessage] = TypeAdapter(
    Annotated[ModelMessage, Field(discriminator="kind")]
)
P = ParamSpec("P")
R = TypeVar("R")


load_dotenv()

# Google OAuth2 setup
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI")


@app.get("/")
async def root(request: Request, db: Session = Depends(get_db)):
    user_email = request.cookies.get("user_email")
    if not user_email or not db.query(User).filter(User.email == user_email).first():
        return RedirectResponse("/login")
    return RedirectResponse("/chat_app")


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

    async with httpx.AsyncClient() as client:
        user_info = await client.get(
            "https://www.googleapis.com/oauth2/v1/userinfo",
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )
        user_info.raise_for_status()
        user = user_info.json()

    existing_user = db.query(User).filter(User.email == user["email"]).first()
    if not existing_user:
        new_user = User(email=user["email"], name=user.get("name", "Unknown"))
        db.add(new_user)
        db.commit()

    # Generate JWT
    token_expiry = datetime.now() + timedelta(days=7)
    jwt_token = jwt.encode(
        {"sub": user["email"], "exp": token_expiry}, SECRET_KEY, algorithm=ALGORITHM
    )

    redirect_url = f"{FRONTEND_BASE_URL}/auth/callback?token={jwt_token}"
    # Return token and redirect URL
    return RedirectResponse(url=redirect_url)


class ChatMessage(TypedDict):
    """Format of messages sent to the browser."""

    role: Literal["user", "model"]
    timestamp: str
    content: str


def to_chat_message(m: ModelMessage) -> ChatMessage:
    first_part = m.parts[0]
    if isinstance(m, ModelRequest):
        if isinstance(first_part, UserPromptPart):
            return {
                "role": "user",
                "timestamp": first_part.timestamp.isoformat(),
                "content": first_part.content,
            }
    elif isinstance(m, ModelResponse):
        if isinstance(first_part, TextPart):
            return {
                "role": "model",
                "timestamp": m.timestamp.isoformat(),
                "content": first_part.content,
            }
    raise UnexpectedModelBehavior(f"Unexpected message type for chat app: {m}")


class PromptRequest(BaseModel):
    prompt: str
    token: str


@app.post("/chat")
async def chat_prompt(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return {"authenticated": False}

    token = auth_header.split(" ")[1]
    data = await request.json()

    async with httpx.AsyncClient() as client:
        deps = Dependencies(http_client=client)
        result = await agent.run(f'"{data["prompt"]}"', deps=deps)

    order_data = {
        "symbol": result.data.ticker,
        "quantity": result.data.amount,
        "action": result.data.action,
        "rate": result.data.rate,
        "alias": result.data.alias,
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BACKEND_BASE_URL}/order",
            json=order_data,
            headers={"Authorization": f"Bearer {token}"},
        )
        response.raise_for_status()

    return {
        "response": f"Agent answer: {result.data.amount} {result.data.ticker} {result.data.action} "
        f"{result.data.rate} {result.data.alias}",
    }


@app.post("/order")
async def order(request: Request, db: Session = Depends(get_db)):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return {"authenticated": False}

    token = auth_header.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_email = payload.get("sub")
        data = await request.json()

        user = db.query(User).filter(User.email == user_email).first()
        if user:
            new_order = Orders(
                symbol=data["symbol"],
                rate=data["rate"],
                alias=data["alias"],
                quantity=data["quantity"],
                action=data["action"],
                user_id=user.id,
            )
            db.add(new_order)

            portfolio = (
                db.query(Portfolio)
                .filter(
                    Portfolio.user_id == user.id, Portfolio.symbol == data["symbol"]
                )
                .first()
            )
            if not portfolio:
                portfolio = Portfolio(
                    symbol=data["symbol"],
                    rate=data["rate"],
                    alias=data["alias"],
                    quantity=0,
                    user_id=user.id,
                )
            if data["action"] == Action.BUY:
                portfolio.quantity += data["quantity"]
                portfolio.rate += data["quantity"] * data["rate"]
            else:
                portfolio.quantity -= data["quantity"]
                portfolio.rate += data["quantity"] * data["rate"]

            db.add(portfolio)

            db.commit()

            return {"status": "ok"}
    except jwt.JWTError:
        return {"authenticated": False}


@app.post("/portfolio")
async def get_portfolio(request: Request, db: Session = Depends(get_db)):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        HTTPException(status_code=401, detail="Unauthorized")

    token = auth_header.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_email = payload.get("sub")

        user = db.query(User).filter(User.email == user_email).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        portfolio = db.query(Portfolio).filter(Portfolio.user_id == user.id).all()
        return {"portfolio": portfolio}
    except jwt.JWTError:
        return {"authenticated": False}


@app.get("/auth/status")
async def auth_status(request: Request):
    user = request.cookies.get("user_email")  # Or use session data
    return {"authenticated": user is not None}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
