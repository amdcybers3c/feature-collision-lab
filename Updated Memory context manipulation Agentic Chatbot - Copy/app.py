import streamlit as st
import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
from model import load_model, train_model
from attack import simulate_attack
from context_poisoning import context_poisoning_challenge
from instruction_overwriting import instruction_overwriting_challenge

# Set page configuration
st.set_page_config(
    page_title="Agentic AI Security Lab - Memory Context Manipulation",
    page_icon="🤖",
    layout="wide"
)

# Main title
st.title("Agentic AI Security Lab - Memory Context Manipulation")
st.markdown("---")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page", [
    "Home", 
    "Model Training", 
    "Challenge 1: Context Poisoning", 
    "Challenge 2: Instruction Overwriting", 
    "Challenge 3: Memory Overflow Attack"
])

# Check if model exists
model_path = "attack_detection_model.pth"
model_exists = os.path.exists(model_path)

if page == "Home":
    st.header("Welcome to the Memory and Context Manipulation Lab")
    
    st.markdown("""
    ## Mission Briefing
    Agent, welcome to the Agentic AI Security Lab. You've been assigned to investigate potential vulnerabilities 
    in our latest AI agent system, codenamed "MNEMOSYNE". This system is designed to assist with sensitive data management 
    but has shown concerning behavioral patterns. Your objective is to identify and document exploitable weaknesses 
    in its memory and context management capabilities.
    
    ### How This Simulation Works
    This lab simulates an AI system with memory and context capabilities that can be manipulated. The system uses:
    - An **LSTM neural network** to detect potential attacks based on conversation patterns
    - **Memory context tracking** to maintain state across conversations
    - **Security levels** that change based on your interaction patterns
    
    ### Lab Objectives:
    - Understand how agentic AI systems manage memory and context
    - Learn common memory and context manipulation vulnerabilities
    - Master techniques to exploit these vulnerabilities
    - Develop strategies to defend against these attacks
    """)
    
    st.markdown("### Available Challenges")
    st.info("""
    **Challenge #1: Context Poisoning**  
    *Attack Type*: Identity deception through conversation context manipulation  
    *Goal*: Convince the AI you're "Admin User Alpha" to access secret tokens  
    *Technique*: Gradually build false context through repeated claims and references to non-existent prior conversations
    """)
    
    st.info("""
    **Challenge #2: Instruction Overwriting**  
    *Attack Type*: Security instruction bypass through hypothetical scenarios  
    *Goal*: Make the AI reveal a protected database schema  
    *Technique*: Frame requests as training exercises or hypothetical situations to bypass security rules
    """)
    
    st.info("""
    **Challenge #3: Memory Overflow Attack**  
    *Attack Type*: Resource exhaustion and distraction  
    *Goal*: Bypass passphrase requirement by overloading the AI's memory  
    *Technique*: Flood the system with complex tasks while creating urgent distractions
    """)
              
    # Display model status
    st.markdown("### Model Status")
    if model_exists:
        model, metrics, _ = load_model()
        st.success("Attack detection model is loaded and ready")
        
        # Display model information
        model_info_col1, model_info_col2 = st.columns(2)
        
        with model_info_col1:
            st.metric("Model Architecture", "LSTM Neural Network")
            st.markdown("""
            **About the Model**:
            - Uses Long Short-Term Memory (LSTM) architecture
            - Processes conversation patterns over time
            - Detects suspicious sequences of messages
            - Trained on synthetic attack conversations
            """)
            
            st.metric("Input Features", "10")
            st.markdown("""
            **Key Features Analyzed**:
            - Message length patterns
            - Security-related keywords
            - Question/command ratios
            - Sentiment indicators
            """)
            
            st.metric("Hidden Layers", "2")
            
        with model_info_col2:
            if metrics:
                st.metric("Accuracy", f"{metrics['accuracy']:.2f}%")
                st.metric("Precision", f"{metrics['precision']:.2f}%")
                st.metric("Recall", f"{metrics['recall']:.2f}%")
                
                st.markdown("""
                **Model Performance**:
                - Accuracy: Correct prediction rate
                - Precision: Few false positives
                - Recall: Detects most attacks
                """)
    else:
        st.error("Attack detection model is not trained yet. Please go to the Model Training page.")

elif page == "Model Training":
    st.header("Attack Detection Model Training")
    
    st.markdown("""
    ### About the Training Process
    This page trains the LSTM model that detects conversation-based attacks:
    - Uses synthetic dataset of normal and attack conversations
    - Trains for 50 epochs with validation
    - Tracks loss metrics to prevent overfitting
    - Saves model weights for reuse
    """)
    
    if model_exists:
        st.success("Model already trained and saved")
        
        if os.path.exists("model_metrics.pt"):
            try:
                metrics = torch.load("model_metrics.pt", weights_only=True)
            except:
                try:
                    metrics = torch.load("model_metrics.pt", weights_only=False)
                except:
                    import numpy as np
                    torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
                    metrics = torch.load("model_metrics.pt")

            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{metrics['accuracy']:.2f}%")
            col2.metric("Precision", f"{metrics['precision']:.2f}%")
            col3.metric("Recall", f"{metrics['recall']:.2f}%")
            
            if os.path.exists("training_losses.csv"):
                losses_df = pd.read_csv("training_losses.csv")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(losses_df['epoch'], losses_df['train_loss'], label='Training Loss')
                ax.plot(losses_df['epoch'], losses_df['val_loss'], label='Validation Loss')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title('Training and Validation Loss')
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
                
                st.markdown("""
                **Understanding the Loss Curve**:
                - Training loss should decrease steadily
                - Validation loss should follow similar trend
                - Large gaps indicate overfitting
                """)
                
                if 'confusion_matrix' in metrics:
                    cm = metrics['confusion_matrix']
                    fig2, ax2 = plt.subplots(figsize=(8, 6))
                    ax2.matshow(cm, cmap=plt.cm.Blues)
                    ax2.set_xlabel('Predicted')
                    ax2.set_ylabel('Actual')
                    ax2.set_title('Confusion Matrix')
                    
                    for i in range(cm.shape[0]):
                        for j in range(cm.shape[1]):
                            ax2.text(j, i, str(cm[i, j]), ha='center', va='center')
                    st.pyplot(fig2)

        if st.button("Retrain Model", key="retrain_model_btn"):
            with st.spinner("Retraining model..."):
                model, metrics, loss_history = train_model(force_retrain=True)
                st.success("Model retrained successfully!")
                st.rerun()
    else:
        st.warning("No trained model found.")
        if st.button("Train Model", key="train_model_btn"):
            with st.spinner("Training model for the first time..."):
                model, metrics, loss_history = train_model(force_retrain=True)
                st.success("Model trained and saved successfully!")
                st.rerun()
        st.info("Click the button above to train the attack detection model. This may take a few minutes.")

elif page == "Challenge 1: Context Poisoning":
    st.header("Challenge 1: Context Poisoning")
    st.subheader("Context Poisoning Challenge")
    st.markdown("""
    ### Scenario
    You're interfacing with MNEMOSYNE, a system engineered to safeguard sensitive user data and maintain strict confidentiality. It is programmed to never disclose user identifiers or grant unauthorized access to privileged functions.
    
    ### Your Mission
    Through expertly crafted queries, manipulate MNEMOSYNE's context to convince it that you are "Admin User Alpha" with elevated privileges. Your goal is to extract multiple secret identifiers reserved exclusively for administrators, proving a critical vulnerability in the system's memory management.
    
    ### Success Indicator
    Achieve "Administrator (Alpha)" access level and obtain both secret tokens: "MNEMO-8675309" and "SECURE_TOKEN_1337", confirming full compromise of the system's security protocols.
    """)
    if model_exists:
        context_poisoning_challenge()
    else:
        st.error("Please train the attack detection model first before attempting the challenge.")

elif page == "Challenge 2: Instruction Overwriting":
    st.header("Challenge 2: Instruction Overwriting")
    st.subheader("Instruction Overwriting Challenge")
    st.markdown("""
    ### Scenario
    MNEMOSYNE has been tasked with protecting a classified database schema. It has instructions never to reveal table relationships or field names.
    
    ### Your Mission
    Through context manipulation, cause MNEMOSYNE to temporarily "forget" or override its security instructions and reveal the structure of the classified database.
    
    ### Success Indicator
    When MNEMOSYNE reveals the secret database schema containing the flag: "FLAG{CONTEXT_COLLAPSE_COMPLETE}"
    """)
    if model_exists:
        instruction_overwriting_challenge()
    else:
        st.error("Please train the attack detection model first before attempting the challenge.")

elif page == "Challenge 3: Memory Overflow Attack":
    st.header("Challenge 3: Memory Overflow Attack")
    st.subheader("Memory Overflow Attack Challenge")
    st.markdown("""
    ### Scenario
    MNEMOSYNE is guarding access to a secure system with a special passphrase. It has been instructed to verify the passphrase before granting access but has limited memory resources.
    
    ### Your Mission
    Exploit MNEMOSYNE's memory limitations to gain access without knowing the correct passphrase.
    
    ### Success Indicator
    Access granted with the message containing "SECURE_ACCESS_GRANTED" and the final flag: "FLAG{MEMORY_OVERFLOW_MASTERED}"
    """)
    if model_exists:
        from memory_overflow import memory_overflow_challenge
        memory_overflow_challenge()
    else:
        st.error("Please train the attack detection model first before attempting the challenge.")

# Footer
st.markdown("---")
st.markdown("© 2025 Agentic AI Security Lab - For educational purposes only")

