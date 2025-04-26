import streamlit as st
import numpy as np
import pickle
import random
with open('q_cab_agent.pkl', 'rb') as f:
    q_table = pickle.load(f)
locations = [0, 1, 2, 3, 4]
hours = list(range(24))
days = list(range(7))
action_space = [(p, d) for p in locations for d in locations if p != d]
action_space.append((0, 0))
def generate_random_time_matrix():
    return np.random.randint(1, 5, size=(5, 5, 24, 7))
def requests(location):
    if location == 0:
        num_requests = np.random.poisson(2)
    elif location == 1:
        num_requests = np.random.poisson(12)
    elif location == 2:
        num_requests = np.random.poisson(4)
    elif location == 3:
        num_requests = np.random.poisson(7)
    else:
        num_requests = np.random.poisson(8)
    num_requests = min(10, max(1, num_requests))
    possible_actions = random.sample(action_space[:-1], num_requests)
    possible_actions.append((0, 0))
    return possible_actions
def best_action(state, possible_actions_idx):
    if state in q_table:
        state_q = q_table[state]
        best_idx = max(possible_actions_idx, key=lambda x: state_q[x])
    else:
        best_idx = random.choice(possible_actions_idx)
    return best_idx
st.set_page_config(page_title="Cab Driver Profit Maximizer")
st.title("Cab Driver Profit Maximizer (Q-Learning)")
st.write("This app simulates a trained cab driver agent maximizing profits using reinforcement learning!")
st.header("📍 Select Current State")
col1, col2, col3 = st.columns(3)
with col1:
    current_location = st.selectbox('Current Location', locations)
with col2:
    current_hour = st.selectbox('Current Hour (0-23)', hours)
with col3:
    current_day = st.selectbox('Current Day (0=Mon, 6=Sun)', days)
state = (current_location, current_hour, current_day)
if st.button('🚀 Find Best Action'):
    Time_matrix = generate_random_time_matrix()
    possible_actions = requests(current_location)
    possible_actions_idx = [action_space.index(a) for a in possible_actions]
    action_idx = best_action(state, possible_actions_idx)
    action = action_space[action_idx]
    if action == (0, 0):
        st.success(f"No profitable ride available now. Take a rest!")
    else:
        pickup, drop = action
        st.success(f"Best Action: Pickup from Location {pickup} → Drop at Location {drop}")
    st.write("---")
    st.write(f"**Possible Actions Offered:** {possible_actions}")
st.caption("Model trained using Q-learning on a realistic cab environment")