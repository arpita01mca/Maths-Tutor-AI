import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

import matplotlib.pyplot as plt
import numpy as np

from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont
import textwrap
import os
import uuid

from moviepy.editor import ImageSequenceClip, AudioFileClip

# ---------------- UI SETUP ----------------
st.set_page_config(page_title="Math Tutor AI", page_icon="🧮")
st.title("🧮 AI Math Tutor")

# ---------------- SIDEBAR ----------------
level = st.sidebar.selectbox(
    "📊 Choose Difficulty Level",
    ["Beginner (Detailed)", "Intermediate", "Exam Mode (Short)"]
)

# ---------------- API KEY (SECURE) ----------------
groq_api_key = st.secrets.get("GROQ_API_KEY", None)

if not groq_api_key:
    st.error("API key not configured. Please contact admin.")
    st.stop()

# ---------------- LLM ----------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=groq_api_key
)

# ---------------- PROMPT ----------------
prompt = ChatPromptTemplate.from_template("""
You are a friendly math tutor.

Solve step-by-step in clear teaching style.

Level: {level}

Format:
1. Understanding
2. Formula
3. Steps
4. Final Answer

Question: {input}
""")

chain = prompt | llm

# ---------------- AUDIO ----------------
def text_to_speech(text):
    file = f"audio_{uuid.uuid4().hex}.mp3"
    tts = gTTS(text=text, lang="en")
    tts.save(file)
    return file

# ---------------- LIMIT SCRIPT ----------------
def limit_script_length(text, max_words=80):
    words = text.split()
    return " ".join(words[:max_words]) + "..." if len(words) > max_words else text

# ---------------- VIDEO CREATION ----------------
def create_video_with_audio(script):
    video_file = "lesson.mp4"
    parts = script.split(". ")

    colors = [
        (52, 152, 219),
        (46, 204, 113),
        (241, 196, 15),
        (231, 76, 60)
    ]

    frame_paths = []

    try:
        font_title = ImageFont.truetype("arial.ttf", 64)
        font_body = ImageFont.truetype("arial.ttf", 42)
    except:
        font_title = ImageFont.load_default()
        font_body = ImageFont.load_default()

    for i, part in enumerate(parts):
        img = Image.new("RGB", (720, 480), colors[i % len(colors)])
        draw = ImageDraw.Draw(img)

        draw.rectangle([0, 0, 720, 120], fill=(0, 0, 0))
        draw.text((20, 30), f"🧮 Step {i+1}", fill=(255, 255, 255), font=font_title)

        draw.rectangle([30, 150, 690, 430], fill=(255, 255, 255))

        wrapped = textwrap.fill(part, width=28)
        draw.text((50, 180), wrapped, fill=(0, 0, 0), font=font_body)

        path = f"frame_{i}.png"
        img.save(path)
        frame_paths.append(path)

    audio_file = text_to_speech(script)
    audio = AudioFileClip(audio_file)

    clip = ImageSequenceClip(frame_paths, fps=len(frame_paths) / audio.duration)
    final = clip.set_audio(audio)

    final.write_videofile(video_file, fps=24)

    for f in frame_paths:
        if os.path.exists(f):
            os.remove(f)

    os.remove(audio_file)

    return video_file

# ---------------- SHORT SCRIPT ----------------
def generate_short_video_script(answer):
    prompt = f"""
    Convert this math solution into a short spoken explanation under 80 words.
    Make it sound like a teacher explaining clearly.

    Solution:
    {answer}
    """
    return llm.invoke(prompt).content

# ---------------- INPUT ----------------
question = st.text_area("✏️ Enter your math question")

col1, col2, col3, col4 = st.columns(4)

with col1:
    solve = st.button("🚀 Solve")

with col2:
    audio_btn = st.button("🔊 Listen")

with col3:
    video_btn = st.button("🎬 Video AI")

with col4:
    hint_btn = st.button("💡 Hint")

# ---------------- SOLVE ----------------
if solve and question:
    with st.spinner("Solving... 🧠"):
        response = chain.invoke({
            "input": question,
            "level": level
        })
        answer = response.content

    st.session_state["last_answer"] = answer

    st.subheader("📘 Explanation")
    st.markdown(answer)

elif solve:
    st.warning("Enter a question first.")

# ---------------- HINT ----------------
if hint_btn and question:
    with st.spinner("Generating hint... 💡"):
        hint_prompt = f"""
        Give ONLY a helpful hint (not full solution) for this math question:

        Question: {question}
        """
        hint = llm.invoke(hint_prompt).content

    st.subheader("💡 Hint")
    st.markdown(hint)

elif hint_btn:
    st.warning("Enter a question first.")

# ---------------- AUDIO ----------------
if audio_btn:
    if "last_answer" in st.session_state:
        audio_file = text_to_speech(st.session_state["last_answer"])
        st.audio(audio_file)
    else:
        st.warning("Solve first.")

# ---------------- VIDEO ----------------
if video_btn:
    if "last_answer" not in st.session_state:
        st.warning("Solve a question first.")
    else:
        with st.spinner("🎬 Creating video..."):
            script = generate_short_video_script(st.session_state["last_answer"])
            script = limit_script_length(script)

            st.write("🎤 Script:", script)

            video_file = create_video_with_audio(script)

            st.success("🎉 Video ready!")
            st.video(video_file)
