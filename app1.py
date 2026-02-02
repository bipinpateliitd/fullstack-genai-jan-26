import streamlit as st

st.title("My First Streamlit App")
st.write("Hello World! Welcome to Streamlit.")
st.markdown("# My First Streamlit App")
st.title("Main Title")
st.header("This is a header")
st.subheader("This is a subheader")
st.text("Plain text")
st.markdown("**Bold**, *Italic*, `Code`")

name = st.text_input("Enter your name")
st.write("Hello", name)

age = st.number_input("Enter age", min_value=0, max_value=100)
st.write("Your age:", age)


if st.button("Click Me"):
    st.success("Button clicked!")


st.sidebar.title("Sidebar Menu")
option = st.sidebar.selectbox(
    "Choose an option",
    ["Home", "About", "Contact"]
)

st.write("You selected:", option)
st.radio("Choose a color", ["Red", "Green", "Blue"])
st.checkbox("Check me")
st.slider("Select a number", 0, 100)
st.selectbox("Choose a fruit", ["Apple", "Banana", "Orange"])
st.multiselect("Choose fruits", ["Apple", "Banana", "Orange"])
st.date_input("Select a date")


st.title("Simple Calculator")

a = st.number_input("Enter first number")
b = st.number_input("Enter second number")

operation = st.selectbox("Operation", ["Add", "Subtract", "Multiply", "Divide"])

if st.button("Calculate"):
    if operation == "Add":
        st.write(a + b)
    elif operation == "Subtract":
        st.write(a - b)
    elif operation == "Multiply":
        st.write(a * b)
    elif operation == "Divide":
        st.write(a / b if b != 0 else "Cannot divide by zero")
col1, col2 = st.columns(2)

with col1:
    st.write("Left Column")

with col2:
    st.write("Right Column")


st.file_uploader("Upload a file")
st.camera_input("Take a photo")
st.color_picker("Pick a color")
st.image("https://www.airforce-technology.com/projects/tejas/")
st.audio("https://via.placeholder.com/150")
st.video("https://youtu.be/4_sPeQSeTdY")
