import json
import os
import re

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
# CONSTANTS
# ============================================================

FAISS_FILE = "nadra_registration_policy.faiss"
METADATA_FILE = "nadra_registration_policy_metadata.json"
CONFIG_FILE = "nadra_registration_policy_config.json"

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "openai/gpt-oss-120b"


# ============================================================
# BASIC STYLING
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

    .source-box {
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #ddd;
        margin-top: 10px;
    }

    .page-badge {
        display: inline-block;
        padding: 4px 9px;
        border-radius: 12px;
        border: 1px solid #aaa;
        margin: 2px;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD CONFIGURATION
# ============================================================

@st.cache_data
def load_config():
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


config = load_config()


# ============================================================
# LOAD FAISS INDEX
# ============================================================

@st.cache_resource
def load_faiss_index():
    return faiss.read_index(FAISS_FILE)


faiss_index = load_faiss_index()


# ============================================================
# LOAD METADATA
# ============================================================

@st.cache_data
def load_metadata():
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


chunk_metadata = load_metadata()


# ============================================================
# NORMALIZE METADATA FORMAT
# ============================================================

if isinstance(chunk_metadata, dict):

    # Common possible formats
    if "chunks" in chunk_metadata:
        chunk_metadata = chunk_metadata["chunks"]

    elif "metadata" in chunk_metadata:
        chunk_metadata = chunk_metadata["metadata"]

    elif "chunk_metadata" in chunk_metadata:
        chunk_metadata = chunk_metadata["chunk_metadata"]

    else:
        # Convert dictionary keyed by vector index
        values = list(chunk_metadata.values())

        if values and isinstance(values[0], dict):
            chunk_metadata = values


# Make sure FAISS index and metadata correspond
if len(chunk_metadata) != faiss_index.ntotal:

    st.error(
        f"FAISS/metadata mismatch: "
        f"FAISS contains {faiss_index.ntotal} vectors, "
        f"but metadata contains {len(chunk_metadata)} records."
    )

    st.stop()


# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model(model_name):
    return SentenceTransformer(model_name)


embedding_model_name = config.get(
    "embedding_model",
    DEFAULT_EMBEDDING_MODEL
)

embedding_model = load_embedding_model(
    embedding_model_name
)


# ============================================================
# DOCUMENT INFORMATION
# ============================================================

DOCUMENT_NAME = config.get(
    "document",
    "Registration Policy"
)

VERSION = config.get(
    "version",
    "RP-6.0.2"
)

EFFECTIVE_DATE = config.get(
    "effective_date",
    "21 Sep 2026"
)

STATUS = config.get(
    "status",
    "Approved"
)


# ============================================================
# GROQ CLIENT
# ============================================================

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")


if not GROQ_API_KEY:
    st.error(
        "GROQ_API_KEY is not configured. "
        "Add it to Streamlit Secrets."
    )
    st.stop()


groq_client = Groq(api_key=GROQ_API_KEY)


LLM_MODEL = config.get(
    "llm_model",
    DEFAULT_LLM_MODEL
)


# ============================================================
# SEARCH HELPERS
# ============================================================

def normalize_for_search(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_heading_only(text):

    if not text:
        return True

    words = text.split()

    if len(words) <= 12:
        return True

    return False


# ============================================================
# QUERY INTENT
# ============================================================

def detect_query_intent(query):

    q = normalize_for_search(query)

    if any(x in q for x in [
        "attestation",
        "attest",
        "who can attest",
        "cnicf"
    ]):
        return "attestation"

    if any(x in q for x in [
        "cancellation",
        "cancel identity",
        "cancel cnic",
        "cancel nicop",
        "death"
    ]):
        return "cancellation"

    if any(x in q for x in [
        "poc",
        "pakistan origin card"
    ]):
        return "poc"

    if any(x in q for x in [
        "date of birth",
        "dob",
        "d o b",
        "birth date",
        "change dob"
    ]):
        return "dob_change"

    if any(x in q for x in [
        "fresh cnic",
        "new cnic",
        "new registration",
        "fresh registration"
    ]):
        return "fresh_registration"

    if any(x in q for x in [
        "renewal",
        "renew",
        "reprint",
        "re print",
        "conversion"
    ]):
        return "conversion"

    return "general"


# ============================================================
# QUERY RETRIEVAL
# ============================================================

def retrieve_policy_chunks(
    query,
    top_k=7
):

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    scores, indices = faiss_index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, index in zip(
        scores[0],
        indices[0]
    ):

        if index < 0:
            continue

        chunk = chunk_metadata[index]

        if isinstance(chunk, dict):
            item = dict(chunk)
        else:
            continue

        item["score"] = float(score)
        item["_index"] = int(index)

        results.append(item)

    return results


# ============================================================
# EVIDENCE ASSEMBLY
# ============================================================

def retrieve_policy_evidence(
    question,
    primary_k=2,
    supporting_k=4
):

    results = retrieve_policy_chunks(
        question,
        top_k=7
    )

    if not results:
        return {
            "question": question,
            "primary_evidence": [],
            "supporting_evidence": [],
            "evidence_text": ""
        }

    primary = results[:primary_k]

    supporting = []

    primary_ids = {
        r.get("chunk_id")
        for r in primary
    }

    primary_pages = {
        r.get("page")
        for r in primary
    }

    # Prefer supporting evidence from different pages
    for r in results[primary_k:]:

        if r.get("chunk_id") in primary_ids:
            continue

        if r.get("page") not in primary_pages:
            supporting.append(r)

        if len(supporting) >= supporting_k:
            break

    # Fill remaining supporting slots
    if len(supporting) < supporting_k:

        supporting_ids = {
            r.get("chunk_id")
            for r in supporting
        }

        for r in results[primary_k:]:

            if r.get("chunk_id") in primary_ids:
                continue

            if r.get("chunk_id") in supporting_ids:
                continue

            supporting.append(r)
            supporting_ids.add(r.get("chunk_id"))

            if len(supporting) >= supporting_k:
                break

    # --------------------------------------------------------
    # BUILD EVIDENCE TEXT
    # --------------------------------------------------------

    evidence_parts = []

    evidence_parts.append(
        f"DOCUMENT: {DOCUMENT_NAME}\n"
        f"VERSION: {VERSION}\n"
        f"STATUS: {STATUS}\n"
        f"EFFECTIVE DATE: {EFFECTIVE_DATE}\n"
    )

    evidence_parts.append(
        f"QUESTION:\n{question}\n"
    )

    evidence_parts.append(
        "PRIMARY EVIDENCE:\n"
    )

    for i, r in enumerate(primary, 1):

        score = r.get(
            "final_score",
            r.get("score", 0)
        )

        evidence_parts.append(
            f"[Primary Evidence {i}]\n"
            f"Page: {r.get('page', 'N/A')}\n"
            f"Section: {r.get('major_section', '')}\n"
            f"Subsection: {r.get('subsection', '')}\n"
            f"Content:\n{r.get('text', '')}\n"
            f"Score: {score:.4f}\n"
        )

    if supporting:

        evidence_parts.append(
            "SUPPORTING EVIDENCE:\n"
        )

        for i, r in enumerate(
            supporting,
            1
        ):

            score = r.get(
                "final_score",
                r.get("score", 0)
            )

            evidence_parts.append(
                f"[Supporting Evidence {i}]\n"
                f"Page: {r.get('page', 'N/A')}\n"
                f"Section: {r.get('major_section', '')}\n"
                f"Subsection: {r.get('subsection', '')}\n"
                f"Content:\n{r.get('text', '')}\n"
                f"Score: {score:.4f}\n"
            )

    return {
        "question": question,
        "primary_evidence": primary,
        "supporting_evidence": supporting,
        "evidence_text": "\n".join(evidence_parts)
    }


# ============================================================
# LLM ANSWER GENERATION
# ============================================================

def generate_policy_answer(question):

    evidence = retrieve_policy_evidence(
        question
    )

    if not evidence["evidence_text"]:
        return {
            "answer": (
                "The provided policy evidence does not "
                "contain enough information to answer "
                "this question."
            ),
            "evidence": evidence
        }

    system_prompt = """
You are a NADRA Registration Policy information assistant.

Answer ONLY from the supplied policy evidence.

STRICT RULES:

1. Do not use outside knowledge.
2. Do not invent requirements, documents, fees,
   procedures or exceptions.
3. If the evidence is insufficient, clearly say so.
4. Preserve conditions such as age and
   resident/non-resident status.
5. Clearly distinguish requirements from remarks.
6. Use simple, practical language.
7. Use bullets or numbered lists where useful.
8. Every factual policy statement must have
   a page citation.
9. Page citations must use exactly:
   [Page X]
10. Do not cite unsupported pages.
11. At the end provide:
   Policy Version
   Effective Date
12. Do not mention FAISS, embeddings, chunks,
   retrieval or internal system details.
"""

    user_prompt = f"""
QUESTION:

{question}

POLICY EVIDENCE:

{evidence["evidence_text"]}

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

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "evidence": evidence
    }


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🇵🇰 NADRA Policy Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    "AI-powered assistant for the NADRA Registration Policy"
    "</div>",
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# DOCUMENT INFORMATION
# ============================================================

with st.expander("📄 Policy Information"):

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Document**")
        st.write(DOCUMENT_NAME)

    with col2:
        st.write("**Version**")
        st.write(VERSION)

    with col3:
        st.write("**Effective Date**")
        st.write(EFFECTIVE_DATE)


# ============================================================
# QUESTION INPUT
# ============================================================

st.subheader("Ask a Policy Question")

question = st.text_area(
    "Enter your question",
    placeholder=(
        "Example: What are the requirements for "
        "changing the date of birth?"
    ),
    height=120
)


ask = st.button(
    "🔎 Ask NADRA Policy Assistant",
    type="primary",
    use_container_width=True
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if ask:

    if not question.strip():

        st.warning(
            "Please enter a policy question."
        )

    else:

        with st.spinner(
            "Searching the Registration Policy and preparing the answer..."
        ):

            try:

                result = generate_policy_answer(
                    question.strip()
                )

                st.subheader("Answer")

                st.markdown(
                    result["answer"]
                )

                st.divider()

                # ------------------------------------------------
                # SOURCE INFORMATION
                # ------------------------------------------------

                st.subheader("📚 Evidence Sources")

                evidence = result["evidence"]

                pages = []

                for r in (
                    evidence["primary_evidence"]
                    + evidence["supporting_evidence"]
                ):

                    page = r.get("page")

                    if page not in pages:
                        pages.append(page)

                if pages:

                    page_text = " ".join(
                        f'<span class="page-badge">Page {p}</span>'
                        for p in pages
                    )

                    st.markdown(
                        page_text,
                        unsafe_allow_html=True
                    )

                st.write(
                    f"**Policy:** {DOCUMENT_NAME}"
                )

                st.write(
                    f"**Version:** {VERSION}"
                )

                st.write(
                    f"**Effective Date:** {EFFECTIVE_DATE}"
                )

                # ------------------------------------------------
                # OPTIONAL EVIDENCE VIEW
                # ------------------------------------------------

                with st.expander(
                    "🔍 View Retrieved Evidence"
                ):

                    for i, r in enumerate(
                        evidence["primary_evidence"],
                        1
                    ):

                        st.markdown(
                            f"### Primary Evidence {i} — "
                            f"Page {r.get('page', 'N/A')}"
                        )

                        st.write(
                            r.get("text", "")
                        )

                    for i, r in enumerate(
                        evidence["supporting_evidence"],
                        1
                    ):

                        st.markdown(
                            f"### Supporting Evidence {i} — "
                            f"Page {r.get('page', 'N/A')}"
                        )

                        st.write(
                            r.get("text", "")
                        )

            except Exception as e:

                st.error(
                    "An error occurred while processing "
                    "your question."
                )

                st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "NADRA Registration Policy Assistant | "
    f"{VERSION} | Effective {EFFECTIVE_DATE}"
)
