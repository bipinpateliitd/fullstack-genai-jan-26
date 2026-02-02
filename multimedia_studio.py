import streamlit as st
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="MediaVerse Studio",
    page_icon="🎬",
    layout="wide"
)

# Header
st.title("🎬 MediaVerse Studio")
st.subheader("Your Ultimate Multimedia Experience Hub")

# Sidebar
with st.sidebar:
    st.header("About MediaVerse")
    st.write("Welcome to MediaVerse Studio - a multimedia platform where you can:")
    st.write("- 📸 Upload and display images")
    st.write("- 🎵 Play audio files")
    st.write("- 🎥 Stream YouTube videos")
    
    st.header("Quick Tips")
    st.info("💡 Use the tabs to switch between different media types")
    st.success("✨ All uploads are processed instantly")

# Main content with tabs
tab1, tab2, tab3 = st.tabs(["📸 Image Gallery", "🎵 Audio Player", "🎥 Video Theater"])

# Tab 1: Image Upload and Display
with tab1:
    st.header("Image Gallery")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Image")
        uploaded_image = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png", "gif", "bmp"],
            key="image_uploader"
        )
        
        st.subheader("Load from Local System")
        local_image_path = st.text_input(
            "Enter image path",
            placeholder="/path/to/your/image.jpg"
        )
        load_local_image = st.button("Load Local Image")
    
    with col2:
        st.subheader("Image Display")
        
        if uploaded_image is not None:
            st.image(uploaded_image, caption=f"Uploaded: {uploaded_image.name}", use_container_width=True)
            st.success(f"Successfully loaded: {uploaded_image.name}")
            file_size = uploaded_image.size / 1024
            st.metric("File Size", f"{file_size:.2f} KB")
            
        elif load_local_image and local_image_path:
            try:
                if Path(local_image_path).exists():
                    st.image(local_image_path, caption=f"Local: {Path(local_image_path).name}", use_container_width=True)
                    st.success(f"Successfully loaded from: {local_image_path}")
                    file_size = Path(local_image_path).stat().st_size / 1024
                    st.metric("File Size", f"{file_size:.2f} KB")
                else:
                    st.error("File not found. Please check the path.")
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
        else:
            st.info("Upload an image or load from local system to display")

# Tab 2: Audio Upload and Play
with tab2:
    st.header("Audio Player")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Audio")
        uploaded_audio = st.file_uploader(
            "Choose an audio file",
            type=["mp3", "wav", "ogg", "m4a"],
            key="audio_uploader"
        )
        
        st.subheader("Load from Local System")
        local_audio_path = st.text_input(
            "Enter audio path",
            placeholder="/path/to/your/audio.mp3"
        )
        load_local_audio = st.button("Load Local Audio")
    
    with col2:
        st.subheader("Audio Playback")
        
        if uploaded_audio is not None:
            st.audio(uploaded_audio)
            st.success(f"Successfully loaded: {uploaded_audio.name}")
            file_size = uploaded_audio.size / 1024
            st.metric("File Size", f"{file_size:.2f} KB")
            
        elif load_local_audio and local_audio_path:
            try:
                if Path(local_audio_path).exists():
                    with open(local_audio_path, 'rb') as audio_file:
                        audio_bytes = audio_file.read()
                        st.audio(audio_bytes)
                        st.success(f"Successfully loaded from: {local_audio_path}")
                        file_size = Path(local_audio_path).stat().st_size / 1024
                        st.metric("File Size", f"{file_size:.2f} KB")
                else:
                    st.error("File not found. Please check the path.")
            except Exception as e:
                st.error(f"Error loading audio: {str(e)}")
        else:
            st.info("Upload an audio file or load from local system to play")

# Tab 3: Video (YouTube)
with tab3:
    st.header("Video Theater")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("YouTube Video")
        youtube_url = st.text_input(
            "Enter YouTube URL",
            placeholder="https://www.youtube.com/watch?v=..."
        )
        
        st.write("**Examples:**")
        st.code("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        st.code("https://youtu.be/dQw4w9WgXcQ")
    
    with col2:
        st.subheader("Video Player")
        
        if youtube_url:
            try:
                # Extract video ID from different YouTube URL formats
                if "youtube.com/watch?v=" in youtube_url:
                    video_id = youtube_url.split("watch?v=")[1].split("&")[0]
                elif "youtu.be/" in youtube_url:
                    video_id = youtube_url.split("youtu.be/")[1].split("?")[0]
                else:
                    video_id = None
                
                if video_id:
                    st.video(f"https://www.youtube.com/watch?v={video_id}")
                    st.success(f"Playing YouTube video: {video_id}")
                else:
                    st.error("Invalid YouTube URL. Please check the format.")
            except Exception as e:
                st.error(f"Error loading video: {str(e)}")
        else:
            st.info("Enter a YouTube URL to start watching")

# Footer
st.divider()
st.write("🎬 MediaVerse Studio - Where Media Comes Alive ✨")
st.caption("Built with ❤️ using Streamlit")
