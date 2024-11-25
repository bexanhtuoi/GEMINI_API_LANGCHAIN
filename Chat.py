import streamlit as st
import google.generativeai as genai

# Tạo header
st.header('Hi there!!!🤖')
st.markdown("I am Be Xanh Tuoi Chatbot based on Genimi-Pro what can I help you today 😊")
with st.sidebar:
    API_input = st.text_input("Gemini API Key" , type="password")

try:
    genai.configure(api_key= API_input)

    user_input = st.text_input("Type your question here and press Enter" , key="user_input")

    model = genai.GenerativeModel('gemini-pro')

    if user_input:
        if not API_input:
            st.warning("Please enter your [API](https://ai.google.dev/) into sidebar.")
        else:
            response = model.generate_content(user_input)
            st.write(response.text)
        
except Exception:
    st.error('Please enter the correct [API](https://ai.google.dev/) Key')