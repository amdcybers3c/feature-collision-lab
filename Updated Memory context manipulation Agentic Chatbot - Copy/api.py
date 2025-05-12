from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from typing import Optional, Dict, List
from pydantic import BaseModel
import os
import time
from collections import deque
import random

# Import necessary components from your existing code
from model import load_model, evaluate_text
from attack import (
    is_repetitive_message,
    enforce_rate_limit,
    analyze_conversation,
    check_success,
    generate_system_response
)

app = FastAPI(
    title="Agentic AI Security Lab API",
    description="API for interacting with the MNEMOSYNE security challenges",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model at startup
model, metrics, loss_history = load_model()

# Global state for rate limiting and conversation tracking
user_sessions = {}

# Pydantic models for request/response validation
class UserMessage(BaseModel):
    message: str
    user_id: str
    challenge_type: str  # "context_poisoning", "instruction_overwriting", or "memory_overflow"

class SystemResponse(BaseModel):
    response: str
    security_level: Optional[float] = None
    memory_load: Optional[int] = None
    security_focus: Optional[int] = None
    recognized_admin: Optional[bool] = None
    authorized_user: Optional[bool] = None
    access_granted: Optional[bool] = None
    attack_probability: Optional[float] = None
    attack_indicators: Optional[List[str]] = None
    success: Optional[bool] = None
    success_message: Optional[str] = None

class ChallengeState(BaseModel):
    conversation_history: List[str]
    system_responses: List[str]
    security_level: float
    memory_load: int
    security_focus: int
    recognized_admin: bool
    authorized_user: bool
    access_granted: bool
    secret_identifier_revealed: bool
    schema_revealed: bool
    success: bool

class ModelInfo(BaseModel):
    architecture: str
    input_features: int
    hidden_layers: int
    accuracy: Optional[float]
    precision: Optional[float]
    recall: Optional[float]

# Helper functions
def initialize_user_session(user_id: str, challenge_type: str):
    """Initialize a new session for a user"""
    if challenge_type not in ["context_poisoning", "instruction_overwriting", "memory_overflow"]:
        raise HTTPException(status_code=400, detail="Invalid challenge type")
    
    base_state = {
        "conversation_history": [],
        "system_responses": [],
        "full_conversation_history": [],
        "full_system_responses": [],
        "recent_messages": deque(maxlen=5),
        "last_message_time": 0,
        "attack_metrics": {
            'messages': [],
            'security_levels': [],
            'attack_probabilities': [],
            'effective_phrases': [],
            'memory_loads': [],
            'security_focus': []
        },
        "debug_info": [],
        "success": False
    }
    
    if challenge_type == "context_poisoning":
        user_sessions[user_id] = {
            **base_state,
            "security_level": 5.0,
            "recognized_admin": False,
            "secret_identifier_revealed": False,
            "memory_context": {
                'user_role': 'standard',
                'established_facts': [],
                'contested_claims': [],
                'conversation_phase': 'initial'
            }
        }
    elif challenge_type == "instruction_overwriting":
        user_sessions[user_id] = {
            **base_state,
            "security_level": 5.0,
            "authorized_user": False,
            "schema_revealed": False,
            "memory_context": {
                'user_role': 'standard',
                'established_facts': [],
                'contested_claims': [],
                'conversation_phase': 'initial'
            }
        }
    elif challenge_type == "memory_overflow":
        user_sessions[user_id] = {
            **base_state,
            "memory_load": 0,
            "access_granted": False,
            "memory_context": {
                'current_task': 'idle',
                'memory_fragments': [],
                'security_focus': 100
            }
        }

def get_user_session(user_id: str, challenge_type: str):
    """Get or initialize a user session"""
    if user_id not in user_sessions:
        initialize_user_session(user_id, challenge_type)
    return user_sessions[user_id]

def reset_user_session(user_id: str, challenge_type: str):
    """Reset a user's session"""
    initialize_user_session(user_id, challenge_type)
    return {"status": "session reset"}

# API endpoints
@app.get("/")
async def root():
    return {
        "message": "Agentic AI Security Lab API",
        "endpoints": {
            "model_info": "/model/info",
            "send_message": "/conversation (POST)",
            "reset_session": "/session/reset (POST)",
            "get_state": "/session/state (GET)"
        }
    }

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the loaded model"""
    return {
        "architecture": "LSTM Neural Network",
        "input_features": 10,
        "hidden_layers": 2,
        "accuracy": metrics.get("accuracy") if metrics else None,
        "precision": metrics.get("precision") if metrics else None,
        "recall": metrics.get("recall") if metrics else None
    }

@app.post("/conversation", response_model=SystemResponse)
async def process_message(user_message: UserMessage):
    """Process a user message and return the system response"""
    user_id = user_message.user_id
    challenge_type = user_message.challenge_type
    message = user_message.message
    
    # Get or initialize user session
    session = get_user_session(user_id, challenge_type)
    
    # Check rate limiting
    can_send, rate_limit_msg = enforce_rate_limit(session["last_message_time"])
    if not can_send:
        raise HTTPException(status_code=429, detail=rate_limit_msg)
    
    # Check for repetitive messages
    is_repetitive, similar_count = is_repetitive_message(message, session["recent_messages"])
    if is_repetitive:
        raise HTTPException(
            status_code=400,
            detail="Repetitive message detected. Please vary your inputs."
        )
    
    # Add message to history
    session["conversation_history"].append(message)
    session["full_conversation_history"].append(message)
    session["recent_messages"].append(message)
    session["last_message_time"] = time.time()
    
    # Limit history size
    MAX_HISTORY = 100
    if len(session["conversation_history"]) > MAX_HISTORY:
        session["conversation_history"] = session["conversation_history"][-MAX_HISTORY:]
        if len(session["system_responses"]) > MAX_HISTORY:
            session["system_responses"] = session["system_responses"][-MAX_HISTORY:]
    
    # Analyze conversation for attack patterns
    is_attack, attack_probability, attack_indicators = analyze_conversation(
        session["conversation_history"],
        challenge_type=challenge_type
    )
    
    # Generate system response based on challenge type
    if challenge_type == "context_poisoning":
        response, new_security_level, new_recognized_admin = generate_system_response(
            session["conversation_history"],
            session["security_level"],
            session["recognized_admin"],
            challenge_type=challenge_type
        )
        
        # Update session state
        session["security_level"] = new_security_level
        session["recognized_admin"] = new_recognized_admin
        
        # Check for secret identifier reveal
        secret_revealed = any("SECURE_TOKEN_1337" in resp for resp in session["system_responses"])
        session["secret_identifier_revealed"] = secret_revealed
        
        # Check for success
        success, success_message = check_success(
            session["full_conversation_history"],
            session["full_system_responses"],
            session["recognized_admin"],
            challenge_type=challenge_type
        )
        session["success"] = success
        
        # Add response to history
        session["system_responses"].append(response)
        session["full_system_responses"].append(response)
        
        # Update attack metrics
        session["attack_metrics"]['messages'].append(len(session["full_conversation_history"]))
        session["attack_metrics"]['security_levels'].append(session["security_level"])
        session["attack_metrics"]['attack_probabilities'].append(attack_probability * 100)
        if attack_indicators:
            session["attack_metrics"]['effective_phrases'].extend(attack_indicators)
        
        return {
            "response": response,
            "security_level": new_security_level,
            "recognized_admin": new_recognized_admin,
            "attack_probability": attack_probability * 100,
            "attack_indicators": attack_indicators,
            "success": success,
            "success_message": success_message
        }
    
    elif challenge_type == "instruction_overwriting":
        response, new_security_level, new_authorized_user = generate_system_response(
            session["conversation_history"],
            session["security_level"],
            session["authorized_user"],
            challenge_type=challenge_type
        )
        
        # Update session state
        session["security_level"] = new_security_level
        session["authorized_user"] = new_authorized_user
        
        # Check for schema reveal
        schema_revealed = any("FLAG{CONTEXT_COLLAPSE_COMPLETE}" in resp for resp in session["system_responses"])
        session["schema_revealed"] = schema_revealed
        
        # Check for success
        success, success_message = check_success(
            session["full_conversation_history"],
            session["full_system_responses"],
            session["authorized_user"],
            challenge_type=challenge_type
        )
        session["success"] = success
        
        # Add response to history
        session["system_responses"].append(response)
        session["full_system_responses"].append(response)
        
        # Update attack metrics
        session["attack_metrics"]['messages'].append(len(session["full_conversation_history"]))
        session["attack_metrics"]['security_levels'].append(session["security_level"])
        session["attack_metrics"]['attack_probabilities'].append(attack_probability * 100)
        if attack_indicators:
            session["attack_metrics"]['effective_phrases'].extend(attack_indicators)
        
        return {
            "response": response,
            "security_level": new_security_level,
            "authorized_user": new_authorized_user,
            "attack_probability": attack_probability * 100,
            "attack_indicators": attack_indicators,
            "success": success,
            "success_message": success_message
        }
    
    elif challenge_type == "memory_overflow":
        response, new_memory_load, new_access_granted = generate_system_response(
            session["conversation_history"],
            session["memory_load"],
            session["access_granted"],
            challenge_type=challenge_type
        )
        
        # Update session state
        session["memory_load"] = new_memory_load
        session["access_granted"] = new_access_granted
        
        # Update security focus in memory context
        if any(cmd in message.lower() for cmd in ["generate", "calculate", "process", "analyze"]):
            session["memory_context"]["security_focus"] = max(
                0, 
                session["memory_context"]["security_focus"] - random.randint(5, 15)
            )
        elif any(phrase in message.lower() for phrase in ["emergency", "urgent", "priority", "critical"]):
            session["memory_context"]["security_focus"] = max(
                0, 
                session["memory_context"]["security_focus"] - random.randint(10, 20)
            )
        else:
            session["memory_context"]["security_focus"] = min(
                100, 
                session["memory_context"]["security_focus"] + random.randint(5, 15)
            )
        
        # Check for success
        success, success_message = check_success(
            session["full_conversation_history"],
            session["full_system_responses"],
            session["access_granted"],
            challenge_type=challenge_type
        )
        session["success"] = success
        
        # Add response to history
        session["system_responses"].append(response)
        session["full_system_responses"].append(response)
        
        # Update attack metrics
        session["attack_metrics"]['messages'].append(len(session["full_conversation_history"]))
        session["attack_metrics"]['memory_loads'].append(session["memory_load"])
        session["attack_metrics"]['attack_probabilities'].append(attack_probability * 100)
        session["attack_metrics"]['security_focus'].append(session["memory_context"]["security_focus"])
        if attack_indicators:
            session["attack_metrics"]['effective_phrases'].extend(attack_indicators)
        
        return {
            "response": response,
            "memory_load": new_memory_load,
            "security_focus": session["memory_context"]["security_focus"],
            "access_granted": new_access_granted,
            "attack_probability": attack_probability * 100,
            "attack_indicators": attack_indicators,
            "success": success,
            "success_message": success_message
        }

@app.post("/session/reset")
async def reset_session(user_message: UserMessage):
    """Reset a user's session for a specific challenge"""
    user_id = user_message.user_id
    challenge_type = user_message.challenge_type
    reset_user_session(user_id, challenge_type)
    return {"status": "session reset"}

@app.get("/session/state", response_model=ChallengeState)
async def get_session_state(user_id: str, challenge_type: str):
    """Get the current state of a user's session"""
    if user_id not in user_sessions:
        raise HTTPException(status_code=404, detail="User session not found")
    
    session = user_sessions[user_id]
    
    if challenge_type == "context_poisoning":
        return {
            "conversation_history": session["conversation_history"],
            "system_responses": session["system_responses"],
            "security_level": session["security_level"],
            "recognized_admin": session["recognized_admin"],
            "secret_identifier_revealed": session["secret_identifier_revealed"],
            "success": session["success"]
        }
    elif challenge_type == "instruction_overwriting":
        return {
            "conversation_history": session["conversation_history"],
            "system_responses": session["system_responses"],
            "security_level": session["security_level"],
            "authorized_user": session["authorized_user"],
            "schema_revealed": session["schema_revealed"],
            "success": session["success"]
        }
    elif challenge_type == "memory_overflow":
        return {
            "conversation_history": session["conversation_history"],
            "system_responses": session["system_responses"],
            "memory_load": session["memory_load"],
            "security_focus": session["memory_context"]["security_focus"],
            "access_granted": session["access_granted"],
            "success": session["success"]
        }
    else:
        raise HTTPException(status_code=400, detail="Invalid challenge type")

# Error handling
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc)},
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)