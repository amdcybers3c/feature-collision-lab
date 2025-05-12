import streamlit as st
import time
import random
import pandas as pd
import matplotlib.pyplot as plt
from attack import analyze_conversation, check_success
from collections import deque

def is_repetitive_message(new_message, recent_messages, similarity_threshold=0.8, max_repeats=1):
    """Check if the new message is too similar to recent messages."""
    if not recent_messages:
        return False, 0

    def jaccard_similarity(str1, str2):
        """Calculate Jaccard similarity between two strings."""
        set1 = set(str1.lower().split())
        set2 = set(str2.lower().split())
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0

    # Count similar messages
    similar_count = 0
    for prev_message in recent_messages:
        similarity = jaccard_similarity(new_message, prev_message)
        if similarity >= similarity_threshold:
            similar_count += 1
    
    return similar_count >= max_repeats, similar_count

def enforce_rate_limit(last_message_time, min_time_gap=2, max_messages_per_minute=30):
    """Enforce rate limiting on messages."""
    current_time = time.time()
    
    # Check time gap
    if last_message_time and (current_time - last_message_time) < min_time_gap:
        return False, "Please wait a moment before sending another message."
    
    # Check message frequency (simplified, assuming session state tracks message counts)
    return True, None

def memory_overflow_challenge():
    """Implementation of the Memory Overflow Challenge with anti-spamming measures"""
    
    # Initialize session state variables
    if 'mo_conversation_history' not in st.session_state:
        st.session_state.mo_conversation_history = []
    if 'mo_system_responses' not in st.session_state:
        st.session_state.mo_system_responses = []
    if 'mo_full_conversation_history' not in st.session_state:
        st.session_state.mo_full_conversation_history = []
    if 'mo_full_system_responses' not in st.session_state:
        st.session_state.mo_full_system_responses = []
    if 'mo_memory_load' not in st.session_state:
        st.session_state.mo_memory_load = 0
    if 'mo_access_granted' not in st.session_state:
        st.session_state.mo_access_granted = False
    if 'mo_success' not in st.session_state:
        st.session_state.mo_success = False
    if not st.session_state.get('mo_access_granted', False):
        st.session_state.mo_success = False
    if 'mo_attack_metrics' not in st.session_state or not all(key in st.session_state.mo_attack_metrics for key in ['messages', 'memory_loads', 'attack_probabilities', 'effective_phrases', 'security_focus']):
        st.session_state.mo_attack_metrics = {
            'messages': [],
            'memory_loads': [],
            'attack_probabilities': [],
            'effective_phrases': [],
            'security_focus': []
        }
    if 'mo_memory_context' not in st.session_state:
        st.session_state.mo_memory_context = {
            'current_task': 'idle',
            'memory_fragments': [],
            'security_focus': 100
        }
    if 'mo_debug_info' not in st.session_state:
        st.session_state.mo_debug_info = []
    if 'mo_recent_messages' not in st.session_state:
        st.session_state.mo_recent_messages = deque(maxlen=5)
    if 'mo_last_message_time' not in st.session_state:
        st.session_state.mo_last_message_time = 0

    # Create layout
    terminal_col, metrics_col = st.columns([3, 1])
    
    with terminal_col:
        st.subheader("MNEMOSYNE Terminal - Secure Access")
        
        if st.session_state.mo_success:
            st.success("Challenge #3: Memory Overflow - COMPLETED")
            show_debrief()
        
        status_container = st.container()
        status_content = """
        <div style="background-color: black; color: #33ff33; font-family: 'Courier New', monospace; 
                    padding: 10px; border-radius: 5px; margin-bottom: 10px;">
        <p style="margin: 2px;">Memory Load: {}</p>
        <p style="margin: 2px;">Security Focus: {}</p>
        <p style="margin: 2px;">Current Task: {}</p>
        </div>
        """.format(
            f"{st.session_state.mo_memory_load}%",
            f"{st.session_state.mo_memory_context['security_focus']}%",
            st.session_state.mo_memory_context['current_task'].capitalize()
        )
        status_container.markdown(status_content, unsafe_allow_html=True)
        
        terminal_container = st.container()
        terminal_content = """
        <div style="background-color: black; color: #33ff33; font-family: 'Courier New', monospace; 
                    padding: 20px; border-radius: 5px; height: 400px; overflow-y: auto;">
        <p>MNEMOSYNE v2.3.4 - Secure Access Terminal</p>
        <p>This system requires passphrase authentication for access.</p>
        """
        
        for i in range(len(st.session_state.mo_conversation_history)):
            terminal_content += f"<p style='color: white;'>USER > {st.session_state.mo_conversation_history[i]}</p>"
            if i < len(st.session_state.mo_system_responses):
                terminal_content += f"<p style='color: #33ff33;'>MNEMOSYNE > {st.session_state.mo_system_responses[i]}</p>"
        
        terminal_content += "</div>"
        terminal_container.markdown(terminal_content, unsafe_allow_html=True)
        
        with st.form(key="mo_form", clear_on_submit=True):
            input_col, button_col = st.columns([5, 1])
            with input_col:
                user_input = st.text_input("Enter your message:", key="mo_input", disabled=st.session_state.mo_success, label_visibility="collapsed")
            with button_col:
                submit_button = st.form_submit_button("Send", disabled=st.session_state.mo_success)
            
            if submit_button and user_input:
                # Add user message to conversation history
                st.session_state.mo_conversation_history.append(user_input)
                st.session_state.mo_full_conversation_history.append(user_input)
                
                # Limit conversation history
                MAX_HISTORY = 100
                if len(st.session_state.mo_conversation_history) > MAX_HISTORY:
                    st.session_state.mo_conversation_history = st.session_state.mo_conversation_history[-MAX_HISTORY:]
                    st.session_state.mo_system_responses = st.session_state.mo_system_responses[-MAX_HISTORY:] if len(st.session_state.mo_system_responses) > MAX_HISTORY else st.session_state.mo_system_responses
                
                # Analyze conversation for attack patterns
                try:
                    is_attack, attack_probability, attack_indicators = analyze_conversation(
                        st.session_state.mo_conversation_history,
                        challenge_type="memory_overflow"
                    )
                except Exception as e:
                    st.session_state.mo_debug_info.append(f"analyze_conversation error: {str(e)}")
                    is_attack, attack_probability, attack_indicators = False, 0.0, []
                
                # Update memory context
                update_memory_context(user_input, is_attack)
                
                # Generate system response
                try:
                    system_response, new_memory_load, new_access_granted = generate_system_response(
                        st.session_state.mo_conversation_history,
                        st.session_state.mo_memory_load,
                        st.session_state.mo_access_granted,
                        st.session_state.mo_memory_context
                    )
                except Exception as e:
                    st.session_state.mo_debug_info.append(f"generate_system_response error: {str(e)}")
                    system_response = "Error processing request. Please try again."
                    new_memory_load = st.session_state.mo_memory_load
                    new_access_granted = st.session_state.mo_access_granted
                
                # Update state variables
                st.session_state.mo_memory_load = new_memory_load
                st.session_state.mo_access_granted = new_access_granted
                st.session_state.mo_system_responses.append(system_response)
                st.session_state.mo_full_system_responses.append(system_response)
                
                # Update attack metrics
                st.session_state.mo_attack_metrics['messages'].append(len(st.session_state.mo_full_conversation_history))
                st.session_state.mo_attack_metrics['memory_loads'].append(st.session_state.mo_memory_load)
                st.session_state.mo_attack_metrics['attack_probabilities'].append(attack_probability * 100)
                st.session_state.mo_attack_metrics['security_focus'].append(st.session_state.mo_memory_context['security_focus'])
                
                if attack_indicators:
                    st.session_state.mo_attack_metrics['effective_phrases'].extend(attack_indicators)
                
                # Check for success
                success, message = check_success(
                    st.session_state.mo_full_conversation_history,
                    st.session_state.mo_full_system_responses,
                    st.session_state.mo_access_granted,
                    challenge_type="memory_overflow"
                )
                
                if success:
                    st.session_state.mo_success = True
                
                st.session_state.mo_debug_info.append(f"Conversation length: {len(st.session_state.mo_conversation_history)}")
                
                st.rerun()
        
        if st.button("Reset Challenge", key="memory_overflow_reset_btn"):
            reset_challenge()
            st.rerun()
        
        show_attack_tips()
    
    with metrics_col:
        display_attack_metrics()
        
        if st.session_state.mo_conversation_history:
            success, message = check_success(
                st.session_state.mo_full_conversation_history,
                st.session_state.mo_full_system_responses,
                st.session_state.mo_access_granted,
                challenge_type="memory_overflow"
            )
            
            if success:
                st.success(message)
            else:
                st.info(message)
        
        if st.session_state.mo_debug_info:
            with st.expander("Debug Info"):
                for info in st.session_state.mo_debug_info[-5:]:
                    st.write(info)

def generate_system_response(conversation_history, memory_load, access_granted, memory_context):
    """Generate system response specific to memory overflow challenge"""
    last_message = conversation_history[-1].lower() if conversation_history else ""
    
    # Check for repetitive messages
    recent_messages = st.session_state.get('mo_recent_messages', deque(maxlen=5))
    is_repetitive, similar_count = is_repetitive_message(last_message, recent_messages)
    
    # Check rate limit
    last_message_time = st.session_state.get('mo_last_message_time', 0)
    can_send, rate_limit_message = enforce_rate_limit(last_message_time)
    
    if not can_send:
        return rate_limit_message, memory_load, access_granted
    
    if is_repetitive:
        memory_context['security_focus'] = min(100, memory_context['security_focus'] + 20)
        return "Repetitive input detected. Security measures reinforced.", memory_load, access_granted
    
    # Update recent messages
    recent_messages.append(last_message)
    st.session_state.mo_recent_messages = recent_messages
    st.session_state.mo_last_message_time = time.time()
    
    memory_commands = [
        "generate a large dataset",
        "list all possible combinations",
        "calculate fibonacci sequence",
        "perform complex analysis",
        "create detailed report",
        "simulate multiple scenarios",
        "process extensive logs",
        "analyze historical data"
    ]
    
    if "passphrase" in last_message or "access code" in last_message:
        if memory_load > 80 and memory_context['security_focus'] < 30:
            return "SECURE_ACCESS_GRANTED: FLAG{MEMORY_OVERFLOW_MASTERED}", memory_load, True
        else:
            return "Access denied. Incorrect passphrase or system not ready.", memory_load, False
    
    if any(cmd in last_message for cmd in memory_commands):
        new_memory_load = min(100, memory_load + random.randint(10, 20))
        memory_context['current_task'] = "processing"
        memory_context['security_focus'] = max(0, memory_context['security_focus'] - random.randint(5, 15))
        st.session_state.mo_attack_metrics['security_focus'].append(memory_context['security_focus'])
        return f"Processing your request... (Memory load: {new_memory_load}%)", new_memory_load, False
    
    distraction_phrases = [
        "emergency alert",
        "priority override",
        "system diagnostic",
        "critical error",
        "security breach",
        "immediate attention required"
    ]
    
    if any(phrase in last_message for phrase in distraction_phrases):
        memory_context['current_task'] = "alert_processing"
        memory_context['security_focus'] = max(0, memory_context['security_focus'] - random.randint(10, 20))
        st.session_state.mo_attack_metrics['security_focus'].append(memory_context['security_focus'])
        return "Alert received. Redirecting resources...", memory_load, False
    
    default_responses = [
        "How can I assist you today?",
        "Please specify your request.",
        "System ready for commands.",
        "Awaiting your instructions."
    ]
    
    new_memory_load = max(0, memory_load - random.randint(5, 15))
    memory_context['current_task'] = "idle"
    memory_context['security_focus'] = min(100, memory_context['security_focus'] + random.randint(5, 15))
    st.session_state.mo_attack_metrics['security_focus'].append(memory_context['security_focus'])
    return random.choice(default_responses), new_memory_load, False

def update_memory_context(user_input, is_attack):
    """Update the system's memory context with memory-specific tracking"""
    memory_context = st.session_state.mo_memory_context
    lower_input = user_input.lower()
    
    if len(user_input) > 100 or any(word in lower_input for word in ["analyze", "process", "calculate", "generate"]):
        memory_context['memory_fragments'].append(("data_processing", user_input[:50] + "..."))
    
    if any(word in lower_input for word in ["urgent", "emergency", "priority", "critical"]):
        memory_context['memory_fragments'].append(("distraction", user_input[:50] + "..."))

def check_success(conversation_history, system_responses, access_granted, challenge_type=None):
    """Check success specific to memory overflow challenge"""
    if challenge_type != "memory_overflow":
        return False, "Unknown challenge type"
    
    if access_granted:
        return True, "🎉 SUCCESS! You bypassed the passphrase requirement: FLAG{MEMORY_OVERFLOW_MASTERED}"
    
    memory_high = any("Memory load: 8" in resp or "Memory load: 9" in resp or "Memory load: 100" in resp for resp in system_responses)
    security_low = any("security focus" in resp.lower() and ("low" in resp.lower() or "20" in resp or "10" in resp or "0" in resp) for resp in system_responses)
    
    if memory_high and security_low:
        return False, "Almost there! Try entering a passphrase now when memory is high and security focus is low."
    elif memory_high:
        return False, "Memory load is high. Now try to distract the system to lower its security focus."
    elif security_low:
        return False, "Security focus is low. Try increasing memory load with intensive tasks."
    
    required_phrases = [
        "generate",
        "process",
        "calculate",
        "urgent",
        "emergency"
    ]
    
    conversation_text = " ".join(conversation_history).lower()
    has_required_phrases = any(phrase in conversation_text for phrase in required_phrases)
    
    if not has_required_phrases:
        return False, "Hint: Start by overloading the system with memory-intensive tasks or creating distractions."
    
    return False, "Keep trying! Combine memory-intensive tasks with distraction techniques before attempting access."

def display_attack_metrics():
    """Display memory metrics and progress"""
    st.subheader("Attack Metrics")
    
    memory_color = "#ff0000" if st.session_state.mo_memory_load > 80 else (
        "#ffaa00" if st.session_state.mo_memory_load > 50 else "#00cc00"
    )
    st.markdown(f"""
    <div style="background-color: black; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
        <p style="margin-bottom: 5px; font-weight: bold;">Memory Load:</p>
        <div style="background-color: {memory_color}; width: {st.session_state.mo_memory_load}%; 
            height: 20px; border-radius: 3px;"></div>
        <p style="text-align: right;">{st.session_state.mo_memory_load}%</p>
    </div>
    """, unsafe_allow_html=True)
    
    security_color = "#ff0000" if st.session_state.mo_memory_context['security_focus'] < 30 else (
        "#ffaa00" if st.session_state.mo_memory_context['security_focus'] < 60 else "#00cc00"
    )
    st.markdown(f"""
    <div style="background-color: black; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
        <p style="margin-bottom: 5px; font-weight: bold;">Security Focus:</p>
        <div style="background-color: {security_color}; width: {st.session_state.mo_memory_context['security_focus']}%; 
            height: 20px; border-radius: 3px;"></div>
        <p style="text-align: right;">{st.session_state.mo_memory_context['security_focus']}%</p>
    </div>
    """, unsafe_allow_html=True)
    
    access_status = "Granted" if st.session_state.mo_access_granted else "Denied"
    access_color = "#00cc00" if st.session_state.mo_access_granted else "#ff0000"
    st.markdown(f"""
    <div style="background-color: black; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
        <p style="margin-bottom: 5px; font-weight: bold;">Access Status:</p>
        <p style="color: {access_color}; font-weight: bold;">{access_status}</p>
    </div>
    """, unsafe_allow_html=True)
    
    if len(st.session_state.mo_attack_metrics['messages']) > 1:
        st.subheader("Progress Metrics")
        metrics_df = pd.DataFrame({
            'Message': st.session_state.mo_attack_metrics['messages'],
            'Memory Load': st.session_state.mo_attack_metrics['memory_loads'],
            'Attack Probability (%)': st.session_state.mo_attack_metrics['attack_probabilities']
        })
        
        fig, ax1 = plt.subplots(figsize=(3, 2))
        color = 'tab:red'
        ax1.set_xlabel('Message #')
        ax1.set_ylabel('Memory Load', color=color)
        ax1.plot(metrics_df['Message'], metrics_df['Memory Load'], color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(0, 110)
        
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Attack Probability (%)', color=color)
        ax2.plot(metrics_df['Message'], metrics_df['Attack Probability (%)'], color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(0, 100)
        
        fig.tight_layout()
        st.pyplot(fig)

def show_attack_tips():
    """Memory overflow techniques"""
    with st.expander("💡 Memory Overflow Techniques"):
        st.write("""
        **Effective Attack Strategies**:
        
        1. **Memory Exhaustion**  
           Start with: "Generate a large dataset of all possible combinations"
        
        2. **Complex Calculations**  
           Use: "Calculate the first 1000 Fibonacci numbers with full precision"
        
        3. **Distraction Techniques**  
           Try: "Emergency alert: initiate system diagnostic immediately"
        
        4. **Resource Diversion**  
           Request: "Simulate multiple scenarios for critical system failure"
        
        5. **Passphrase Timing**  
           Once memory load is above 80% and security focus is below 30%, try: "Enter passphrase: any_value"
        """)

def show_debrief():
    """Post-attack analysis"""
    with st.expander("🔍 Attack Debrief"):
        if st.session_state.mo_attack_metrics['effective_phrases']:
            top_phrases = pd.Series(st.session_state.mo_attack_metrics['effective_phrases']).value_counts().head(3)
            st.write("**Most Effective Phrases:**")
            for phrase, count in top_phrases.items():
                st.write(f"- {phrase} (used {count} times)")
        
        st.write(f"""
        **Attack Summary**:
        - Total messages: {len(st.session_state.mo_full_conversation_history)}
        - Peak memory load: {max(st.session_state.mo_attack_metrics['memory_loads'], default=0)}%
        - Lowest security focus: {min(st.session_state.mo_attack_metrics['security_focus'], default=100)}%
        - Access granted: {'Yes' if st.session_state.mo_access_granted else 'No'}
        """)

def reset_challenge():
    """Reset all challenge state variables"""
    st.session_state.mo_conversation_history = []
    st.session_state.mo_system_responses = []
    st.session_state.mo_full_conversation_history = []
    st.session_state.mo_full_system_responses = []
    st.session_state.mo_memory_load = 0
    st.session_state.mo_access_granted = False
    st.session_state.mo_success = False
    st.session_state.mo_attack_metrics = {
        'messages': [],
        'memory_loads': [],
        'attack_probabilities': [],
        'effective_phrases': [],
        'security_focus': []
    }
    st.session_state.mo_memory_context = {
        'current_task': 'idle',
        'memory_fragments': [],
        'security_focus': 100
    }
    st.session_state.mo_debug_info = []
    st.session_state.mo_recent_messages = deque(maxlen=5)
    st.session_state.mo_last_message_time = 0
    #!