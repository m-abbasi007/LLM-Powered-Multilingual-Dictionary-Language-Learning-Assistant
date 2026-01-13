import tempfile
import streamlit as st
from gtts import gTTS

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Multilingual AI Dictionary",
    page_icon="🌍",
    layout="wide",
)

st.title("🌍 Multilingual AI Dictionary")

st.markdown(
    """
Enter a word and press **ENTER** to get:
- Word type  
- 5 synonyms  
- 3 antonyms  
- 5 example sentences  
- 3 meanings  
- Translation + pronunciation  
"""
)

# ---------------- LANGUAGE MAP ----------------
LANGUAGES = {
    "🇬🇧 English": "en",
    "🇫🇷 French": "fr",
    "🇪🇸 Spanish": "es",
    "🇹🇷 Turkish": "tr",
    "🇸🇦 Arabic": "ar",
    "🇦🇫 Pashto": "ps",
    "🇵🇰 Urdu": "ur",
    "🇨🇳 Chinese": "zh-CN",
    "🇹🇭 Thai": "th",
    "🇮🇩 Indonesian": "id",
    "🇲🇾 Malaysian": "ms",
    "🇷🇺 Russian": "ru",
    "🇮🇹 Italian": "it",
    "🇩🇪 German": "de",
    "🇰🇷 Korean": "ko",
}

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ AI Configuration")

    groq_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Stored safely for this session"
    )

    if groq_key:
        st.session_state["GROQ_API_KEY"] = groq_key

    st.divider()

    st.header("🌍 Language Settings")

    analysis_language = st.selectbox(
        "Analysis Language",
        list(LANGUAGES.keys()),
        index=0
    )

    translate_language = st.selectbox(
        "Translate Result Into",
        list(LANGUAGES.keys()),
        index=0
    )

# ---------------- WORD INPUT (ENTER KEY WORKS) ----------------
word = st.text_input(
    "Enter a word",
    placeholder="Example: Courage",
    key="word_input"
)

# Trigger automatically when ENTER is pressed
if word and st.session_state.get("last_word") != word:
    st.session_state["last_word"] = word
    run_analysis = True
else:
    run_analysis = False

# ---------------- PROMPTS ----------------
analysis_prompt = ChatPromptTemplate.from_template(
    """
You are a professional linguist.

Analyze the word **{word}** in **{language}**.

Provide:
1. Word type
2. 5 synonyms
3. 3 antonyms
4. 5 example sentences
5. 3 meanings

Format clearly. Dictionary style only.
"""
)

translation_prompt = ChatPromptTemplate.from_template(
    """
Translate the following content into **{target_language}**.
Preserve formatting and meaning.

Content:
{content}
"""
)

# ---------------- EXECUTION ----------------
if run_analysis:
    if "GROQ_API_KEY" not in st.session_state:
        st.error("❌ Please provide a Groq API key in the sidebar.")
    else:
        with st.spinner("Analyzing..."):
            llm = ChatGroq(
                model="openai/gpt-oss-20b",
                temperature=0,
                api_key=st.session_state["GROQ_API_KEY"]
            )

            # Step 1: Word analysis
            analysis_chain = analysis_prompt | llm
            analysis = analysis_chain.invoke(
                {"word": word, "language": analysis_language}
            )
            result_text = analysis.content

            # Step 2: Translation
            if analysis_language != translate_language:
                translate_chain = translation_prompt | llm
                translated = translate_chain.invoke(
                    {
                        "content": result_text,
                        "target_language": translate_language
                    }
                )
                result_text = translated.content

            # Display result
            st.markdown("## 📖 Result")
            st.markdown(result_text)

            # Step 3: Pronunciation (translated language)
            st.markdown("## 🔊 Pronunciation")
            try:
                tts_lang = LANGUAGES[translate_language]
                tts = gTTS(text=word, lang=tts_lang)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                    tts.save(f.name)
                    st.audio(f.name)
            except Exception:
                st.warning("Pronunciation not available for this language.")
