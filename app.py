import streamlit as st



st.title("GENAI-APP")


st.write("This is a simple app to demonstrate the use of GENAI")


text="""

Robotics is basically the science (and art 😄) of **building machines that can sense, think, and act** in the real world. These machines—**robots**—are designed to help humans, automate work, or do things that are dangerous, repetitive, or super precise.

"""

st.markdown(text)


st.markdown("### Input the text")

input_text=st.text_area("Input the text")


st.markdown("### Output")

st.write(input_text)
st.button("Submit")
st.radio("Select the language", ["English", "Hindi"])

st.selectbox("Select the language", ["English", "Hindi","Gujarati","Marathi","Tamil","Telugu","Kannada","Odiya","Punjabi","Bengali"])

st.slider("Select the number", 1, 100)
st.sidebar.title("GENAI MODELS")
st.sidebar.selectbox("Select the model", ["GPT-3", "GPT-4", "GPT-4o", "GPT-4o-mini"])

st.sidebar.button("Submit")
st.sidebar.number_input("Select the number", 1, 100)

st.video("https://youtu.be/KgOqjXsTf1w")

st.header("About")
st.markdown("This is a simple app to demonstrate the use of GENAI")
st.subheader("Introduction")
st.markdown("This is a simple app to demonstrate the use of GENAI")
st.success("This is a simple app to demonstrate the use of GENAI")
st.error("This is a simple app to demonstrate the use of GENAI")
st.warning("This is a simple app to demonstrate the use of GENAI")
st.info("This is a simple app to demonstrate the use of GENAI")
st.badge("This is a simple app to demonstrate the use of GENAI")
st.caption("This is a simple app to demonstrate the use of GENAI")
st.code("a=43  b=53\nc=a+b")

st.file_uploader("Upload a  .wav file")
st.feedback("faces")
st.toggle("Activate feature")
st.chat_input("Type your message here...")
st.chat_message("user")
st.chat_message("assistant")
st.date_input("Select the date")
st.time_input("Select the time")
