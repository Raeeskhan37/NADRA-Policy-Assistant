import json
import os
import pickle

import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NADRA Policy Assistant",
    page_icon="🇵🇰",
    layout="wide"
)


# ============================================================
# FILE PATHS
# ============================================================

ENGLISH_FAISS = "nadra_registration_policy.faiss"
ENGLISH_METADATA = "nadra_registration_policy_metadata.json"
ENGLISH_CONFIG = "nadra_registration_policy_config.json"

URDU_FAISS = "urdu/nadra_urdu_6_0_2_v2.faiss"
URDU_CHUNKS = "urdu/nadra_urdu_6_0_2_v2_chunks.pkl"
URDU_CONFIG = "urdu/metadata.json"


# ============================================================
# MODELS
# ============================================================

ENGLISH_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
URDU_EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

DEFAULT_LLM_MODEL = "openai/gpt-oss-120b"


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 34px;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
        margin-top: 0;
    }

    .page-badge {
        display: inline-block;
        padding: 4px 9px;
        border-radius: 12px;
        border: 1px solid #aaa;
        margin: 2px;
        font-size: 13px;
    }

    .urdu-text {
        direction: rtl;
        text-align: right;
        font-size: 18px;
        line-height: 2;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LANGUAGE SELECTION
# ============================================================

language = st.radio(
    "Language / زبان",
    ["English", "اردو"],
    horizontal=True
)

is_urdu = language == "اردو"


# ============================================================
# LOAD ENGLISH CONFIG
# ============================================================

@st.cache_data
def load_english_config():

    with open(
        ENGLISH_CONFIG,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


english_config = load_english_config()


# ============================================================
# LOAD URDU CONFIG
# ============================================================

@st.cache_data
def load_urdu_config():

    with open(
        URDU_CONFIG,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


urdu_config = load_urdu_config()


# ============================================================
# LOAD ENGLISH FAISS
# ============================================================

@st.cache_resource
def load_english_index():

    return faiss.read_index(
        ENGLISH_FAISS
    )


# ============================================================
# LOAD URDU FAISS
# ============================================================

@st.cache_resource
def load_urdu_index():

    return faiss.read_index(
        URDU_FAISS
    )


# ============================================================
# LOAD ENGLISH METADATA
# ============================================================

@st.cache_data
def load_english_metadata():

    with open(
        ENGLISH_METADATA,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    if isinstance(data, dict):

        if "chunks" in data:
            data = data["chunks"]

        elif "metadata" in data:
            data = data["metadata"]

        elif "chunk_metadata" in data:
            data = data["chunk_metadata"]

        else:

            values = list(data.values())

            if values and isinstance(
                values[0],
                dict
            ):
                data = values

    return data


# ============================================================
# LOAD URDU CHUNKS
# ============================================================

@st.cache_data
def load_urdu_chunks():

    with open(
        URDU_CHUNKS,
        "rb"
    ) as f:

        data = pickle.load(f)

    return data


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_model(model_name):

    return SentenceTransformer(
        model_name
    )


# ============================================================
# SELECT ACTIVE DATASET
# ============================================================

if is_urdu:

    faiss_index = load_urdu_index()

    chunk_metadata = load_urdu_chunks()

    embedding_model = load_model(
        URDU_EMBEDDING_MODEL
    )

    document_name = urdu_config.get(
        "document",
        "NADRA Registration Policy 6.0.2"
    )

    version = "6.0.2"

    effective_date = "21 Sep 2026"

    status = "Approved"

else:

    faiss_index = load_english_index()

    chunk_metadata = load_english_metadata()

    embedding_model = load_model(
        english_config.get(
            "embedding_model",
            ENGLISH_EMBEDDING_MODEL
        )
    )

    document_info = english_config.get(
        "document",
        {}
    )

    if isinstance(
        document_info,
        dict
    ):

        document_name = document_info.get(
            "document",
            "Registration Policy"
        )

        version = document_info.get(
            "version",
            "RP-6.0.2"
        )

        effective_date = document_info.get(
            "effective_date",
            "21 Sep 2026"
        )

        status = document_info.get(
            "status",
            "Approved"
        )

    else:

        document_name = "Registration Policy"
        version = "RP-6.0.2"
        effective_date = "21 Sep 2026"
        status = "Approved"


# ============================================================
# VALIDATE DATA
# ============================================================

if len(chunk_metadata) != faiss_index.ntotal:

    st.error(
        f"FAISS/chunk mismatch. "
        f"Index contains {faiss_index.ntotal} vectors "
        f"but {len(chunk_metadata)} chunk records were loaded."
    )

    st.stop()


# ============================================================
# GROQ
# ============================================================

try:

    GROQ_API_KEY = st.secrets[
        "GROQ_API_KEY"
    ]

except Exception:

    GROQ_API_KEY = os.environ.get(
        "GROQ_API_KEY"
    )


if not GROQ_API_KEY:

    st.error(
        "GROQ_API_KEY is not configured."
    )

    st.stop()


groq_client = Groq(
    api_key=GROQ_API_KEY
)


LLM_MODEL = DEFAULT_LLM_MODEL


# ============================================================
# TEXT EXTRACTION
# ============================================================

def get_chunk_text(chunk):

    if isinstance(
        chunk,
        str
    ):
        return chunk

    if isinstance(
        chunk,
        dict
    ):

        for key in [
            "text",
            "content",
            "chunk",
            "page_text"
        ]:

            if key in chunk:

                return str(
                    chunk[key]
                )

    return str(chunk)


# ============================================================
# PAGE EXTRACTION
# ============================================================

def get_page(chunk):

    if isinstance(
        chunk,
        dict
    ):

        return chunk.get(
            "page",
            "N/A"
        )

    return "N/A"


# ============================================================
# SECTION EXTRACTION
# ============================================================

def get_section(chunk):

    if isinstance(
        chunk,
        dict
    ):

        return (
            chunk.get(
                "major_section",
                ""
            )
            or chunk.get(
                "section",
                ""
            )
        )

    return ""


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_chunks(
    question,
    top_k=6
):

    query = question

    # E5 models work better when the query
    # is explicitly marked as a query.
    if is_urdu:

        query = "query: " + question

    embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )

    embedding = np.asarray(
        embedding,
        dtype="float32"
    )

    scores, indices = faiss_index.search(
        embedding,
        min(
            top_k,
            faiss_index.ntotal
        )
    )

    results = []

    for score, index in zip(
        scores[0],
        indices[0]
    ):

        if index < 0:
            continue

        chunk = chunk_metadata[index]

        results.append(
            {
                "text": get_chunk_text(
                    chunk
                ),
                "page": get_page(
                    chunk
                ),
                "section": get_section(
                    chunk
                ),
                "score": float(
                    score
                ),
                "index": int(
                    index
                )
            }
        )

    return results


# ============================================================
# BUILD EVIDENCE
# ============================================================

def build_evidence(
    question,
    results
):

    parts = []

    parts.append(
        f"DOCUMENT: {document_name}"
    )

    parts.append(
        f"VERSION: {version}"
    )

    parts.append(
        f"STATUS: {status}"
    )

    parts.append(
        f"EFFECTIVE DATE: {effective_date}"
    )

    parts.append(
        f"QUESTION: {question}"
    )

    parts.append(
        "POLICY EVIDENCE:"
    )

    for i, item in enumerate(
        results,
        1
    ):

        parts.append(
            f"""
[EVIDENCE {i}]
Page: {item["page"]}
Section: {item["section"]}
Similarity: {item["score"]:.4f}

Content:
{item["text"]}
"""
        )

    return "\n".join(
        parts
    )


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question
):

    results = retrieve_chunks(
        question,
        top_k=6
    )

    if not results:

        return (
            "No relevant policy evidence was found.",
            results
        )

    evidence = build_evidence(
        question,
        results
    )

    if is_urdu:

        system_prompt = """
آپ NADRA Registration Policy کے معلوماتی معاون ہیں۔

جواب صرف فراہم کردہ پالیسی کے شواہد کی بنیاد پر دیں۔

اہم اصول:

1. بیرونی معلومات استعمال نہ کریں۔
2. کوئی شرط، دستاویز، فیس، طریقہ کار یا استثنا خود سے نہ بنائیں۔
3. اگر شواہد کافی نہ ہوں تو واضح طور پر بتائیں۔
4. پالیسی کی شرائط کو درست طور پر بیان کریں۔
5. آسان اور واضح اردو استعمال کریں۔
6. جہاں مناسب ہو بلٹس یا نمبر وار فہرست استعمال کریں۔
7. ہر اہم پالیسی دعوے کے ساتھ [Page X] لکھیں۔
8. صرف فراہم کردہ شواہد کے صفحات کو cite کریں۔
9. آخر میں Policy Version اور Effective Date دیں۔
10. FAISS، embeddings، chunks یا اندرونی تکنیکی نظام کا ذکر نہ کریں۔
"""

        user_prompt = f"""
سوال:

{question}

پالیسی کے شواہد:

{evidence}

صرف اوپر دیے گئے شواہد کی بنیاد پر اردو میں جواب دیں۔
متعلقہ معلومات کے ساتھ [Page X] citation ضرور دیں۔
"""

    else:

        system_prompt = """
You are a NADRA Registration Policy information assistant.

Answer ONLY from the supplied policy evidence.

STRICT RULES:

1. Do not use outside knowledge.
2. Do not invent requirements, documents, fees,
   procedures or exceptions.
3. If the evidence is insufficient, clearly say so.
4. Preserve policy conditions accurately.
5. Use simple, practical language.
6. Use bullets or numbered lists where useful.
7. Every factual policy statement should have
   a [Page X] citation.
8. Do not cite unsupported pages.
9. At the end provide:
   Policy Version
   Effective Date
10. Do not mention FAISS, embeddings, chunks,
    retrieval or internal system details.
"""

        user_prompt = f"""
QUESTION:

{question}

POLICY EVIDENCE:

{evidence}

Answer the question using ONLY the evidence above.

Include [Page X] citations after relevant statements.
"""

    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.1,
        max_tokens=2500
    )

    answer = response.choices[
        0
    ].message.content

    return answer, results


# ============================================================
# HEADER
# ============================================================

if is_urdu:

    st.markdown(
        '<div class="main-title">🇵🇰 نادرا پالیسی اسسٹنٹ</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'نادرا رجسٹریشن پالیسی کے لیے معلوماتی معاون'
        '</div>',
        unsafe_allow_html=True
    )

else:

    st.markdown(
        '<div class="main-title">'
        '🇵🇰 NADRA Policy Assistant'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'AI-powered assistant for the NADRA Registration Policy'
        '</div>',
        unsafe_allow_html=True
    )


st.divider()


# ============================================================
# POLICY INFORMATION
# ============================================================

with st.expander(
    "📄 Policy Information"
    if not is_urdu
    else "📄 پالیسی کی معلومات"
):

    col1, col2, col3 = st.columns(3)

    with col1:

        st.write(
            "**Document**"
            if not is_urdu
            else "**دستاویز**"
        )

        st.write(
            document_name
        )

    with col2:

        st.write(
            "**Version**"
            if not is_urdu
            else "**ورژن**"
        )

        st.write(
            version
        )

    with col3:

        st.write(
            "**Effective Date**"
            if not is_urdu
            else "**مؤثر تاریخ**"
        )

        st.write(
            effective_date
        )


# ============================================================
# QUESTION
# ============================================================

if is_urdu:

    st.subheader(
        "پالیسی سے متعلق سوال پوچھیں"
    )

    question = st.text_area(
        "اپنا سوال درج کریں",
        placeholder=(
            "مثال: تاریخ پیدائش تبدیل کرنے "
            "کے لیے کیا شرائط ہیں؟"
        ),
        height=120
    )

    ask_label = (
        "🔎 نادرا پالیسی سے جواب حاصل کریں"
    )

else:

    st.subheader(
        "Ask a Policy Question"
    )

    question = st.text_area(
        "Enter your question",
        placeholder=(
            "Example: What are the requirements "
            "for changing the date of birth?"
        ),
        height=120
    )

    ask_label = (
        "🔎 Ask NADRA Policy Assistant"
    )


ask = st.button(
    ask_label,
    type="primary",
    use_container_width=True
)


# ============================================================
# PROCESS
# ============================================================

if ask:

    if not question.strip():

        if is_urdu:

            st.warning(
                "براہ کرم پالیسی سے متعلق سوال درج کریں۔"
            )

        else:

            st.warning(
                "Please enter a policy question."
            )

    else:

        with st.spinner(
            "جواب تیار کیا جا رہا ہے..."
            if is_urdu
            else
            "Searching the policy and preparing the answer..."
        ):

            try:

                answer, results = generate_answer(
                    question.strip()
                )

                if is_urdu:

                    st.subheader(
                        "جواب"
                    )

                    st.markdown(
                        f'<div class="urdu-text">{answer}</div>',
                        unsafe_allow_html=True
                    )

                else:

                    st.subheader(
                        "Answer"
                    )

                    st.markdown(
                        answer
                    )

                st.divider()

                if is_urdu:

                    st.subheader(
                        "📚 حوالہ جات"
                    )

                else:

                    st.subheader(
                        "📚 Evidence Sources"
                    )

                pages = []

                for result in results:

                    page = result.get(
                        "page"
                    )

                    if page not in pages:

                        pages.append(
                            page
                        )

                if pages:

                    page_text = " ".join(
                        f'<span class="page-badge">'
                        f'Page {p}'
                        f'</span>'
                        for p in pages
                    )

                    st.markdown(
                        page_text,
                        unsafe_allow_html=True
                    )

                st.write(
                    f"**Policy:** {document_name}"
                )

                st.write(
                    f"**Version:** {version}"
                )

                st.write(
                    f"**Effective Date:** {effective_date}"
                )

                with st.expander(
                    "🔍 View Retrieved Evidence"
                    if not is_urdu
                    else
                    "🔍 حاصل شدہ پالیسی شواہد دیکھیں"
                ):

                    for i, result in enumerate(
                        results,
                        1
                    ):

                        st.markdown(
                            f"### Evidence {i} — "
                            f"Page {result.get('page', 'N/A')}"
                        )

                        st.write(
                            result.get(
                                "text",
                                ""
                            )
                        )

            except Exception as e:

                st.error(
                    "An error occurred while processing "
                    "your question."
                    if not is_urdu
                    else
                    "سوال پر کارروائی کے دوران خرابی پیش آئی۔"
                )

                st.exception(
                    e
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "NADRA Policy Assistant | "
    f"{version} | Effective {effective_date}"
)
