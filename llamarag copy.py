import streamlit as st
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
import fitz  
import re

# --- CONFIGURATION ---
st.set_page_config(layout="centered")
st.title("Customer Support Assistant - Llama Technologies 🦙")

API_KEY = "......................................"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

DATA_PATH = "Customer_Support_Dataset.csv"
EMBED_DIM = 384 

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_concern" not in st.session_state:
    st.session_state.selected_concern = None
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "context_df" not in st.session_state:
    st.session_state.context_df = None

# --- LOAD AND PREPROCESS DATA ---
@st.cache_resource
def load_and_index_data():
    df = pd.read_csv(DATA_PATH)

    # Standardize column names
    if 'instruction' in df.columns and 'response' in df.columns:
        df = df.rename(columns={"instruction": "prompt"})

    df = df.dropna(subset=["prompt", "response"])
    df = df.sample(n=min(200, len(df)), random_state=42)

    # Embedding
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    embeddings = embedder.encode(df["prompt"].tolist(), show_progress_bar=True)

    # FAISS Index
    index = faiss.IndexFlatL2(EMBED_DIM)
    index.add(np.array(embeddings).astype("float32"))

    return index, df, embedder

faiss_index, context_df, embedder = load_and_index_data()
st.session_state.faiss_index = faiss_index
st.session_state.context_df = context_df

# --- CONCERNS ---
concerns = [
    "cancel_order", "change_order", "change_shipping_address", "check_payment_methods", "complaint",
    "contact_customer_service", 
    "delivery_options", "delivery_period",  "get_refund",
    "payment_issue", "place_order", "set_up_shipping_address", 
    "track_order", "track_refund"
]

# --- INITIAL PROMPT ---
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown("Hi, I'm the customer support chatbot for Llama Technologies 🦙🤖. I am happy to assist you with necessary information. Tell me what your concern is about:")

# --- CONCERN BUTTONS ---
if not st.session_state.selected_concern:
    cols = st.columns(5)
    for i, concern in enumerate(concerns):
        with cols[i % 5]:
            if st.button(concern.replace("_", " ").capitalize()):
                st.session_state.selected_concern = concern
                st.session_state.messages.append({
                    "role": "user",
                    "content": concern.replace("_", " ").capitalize()
                })
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"I get that you are concerned with **{concern.replace('_', ' ')}**. If you have the invoice related to your concern, please upload it and enter your question."
                })
                st.rerun()


# --- UPLOAD INVOICE ---
uploaded_invoice = st.file_uploader("Upload your invoice (PDF, 1 page)", type=["pdf"])
invoice_text = ""

# --- EXTRACT INVOICE DATA ---
def extract_text_from_pdf(file):
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text.strip()

def parse_invoice_text(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    info = {
        "Customer Name": "",
        "Shipping Address": "",
        "Shipping Method": "",
        "Order Date": "",
        "Order Number": "",
        "Items": [],
        "Total": ""
    }

    i = 0
    while i < len(lines):
        line = lines[i]

        if line.startswith("Customer Name:"):
            info["Customer Name"] = line.split("Customer Name:")[1].strip()
        elif line.startswith("Shipping Address:"):
            info["Shipping Address"] = line.split("Shipping Address:")[1].strip()
        elif line.startswith("Shipping Method:"):
            info["Shipping Method"] = line.split("Shipping Method:")[1].strip()
        elif line.startswith("Order Date:"):
            info["Order Date"] = line.split("Order Date:")[1].strip()
        elif line.startswith("Order Number:"):
            info["Order Number"] = line.split("Order Number:")[1].strip()
        elif line == "Item":
            # skip headers: Quantity, Price, Total
            i += 4
            items = []
            while i + 3 < len(lines) and not lines[i].startswith("Total"):
                item = lines[i]
                quantity = lines[i+1]
                price = lines[i+2]
                line_total = lines[i+3]
                items.append({
                    "Item": item,
                    "Quantity": quantity,
                    "Price": price,
                    "Total": line_total
                })
                i += 4
            info["Items"] = items
            continue
        elif line == "Overall Total" and i + 1 < len(lines):
            info["Total"] = lines[i+1]
            break
        i += 1

    return info


if uploaded_invoice:
    invoice_text = extract_text_from_pdf(uploaded_invoice)
    parsed_data = parse_invoice_text(invoice_text)

    st.success("Invoice uploaded and processed successfully.")
    st.subheader("📄 Invoice Summary")

    st.write(f"**Customer Name:** {parsed_data['Customer Name']}")
    st.write(f"**Shipping Address:** {parsed_data['Shipping Address']}")
    st.write(f"**Shipping Method:** {parsed_data['Shipping Method']}")
    st.write(f"**Order Date:** {parsed_data['Order Date']}")
    st.write(f"**Order Number:** {parsed_data['Order Number']}")

    if parsed_data["Items"]:
        st.markdown("### 🧾 Items Ordered")
        df = pd.DataFrame(parsed_data["Items"])
        st.dataframe(df)

    st.write(f"**Total Amount:** {parsed_data['Total']}")


# --- CHAT INPUT ---
user_input = st.chat_input("Ask your question here...")

# --- RETRIEVAL AUGMENTED GENERATION ---
def retrieve_context(query, k=3):
    query_embedding = embedder.encode([query])
    _, indices = faiss_index.search(np.array(query_embedding).astype("float32"), k)
    responses = context_df.iloc[indices[0]]["response"].tolist()
    return "\n\n".join(responses)

if user_input and API_KEY:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("Thinking..."):
        try:
            # Retrieve relevant context
            context = retrieve_context(user_input)

            print(context)

            # Add context to system message
            system_prompt = f"""You are a helpful customer support assistant for a company named Llama Technologies.
            You have to answer the questions asked by the user regarding concerns in the {concerns}.
            Use the following context to answer: \n\n{context}

            We offer 4 delivery options:
                Standard Shipping (6-8 business days)
                Expedited Shipping (2-3 business days)
                Overnight Shipping (next business day)
                In-Store Pickup (Pick up at the store nearest to you)"""

            if invoice_text:
                system_prompt += f"\n\nThe following is additional information extracted from the user's uploaded invoice:\n{invoice_text}\n\nUse this to personalize your response if relevant."

            system_prompt += """
            Output format:
            -Give short answers. Reduce explanations.
            -Reject answering questions unrelated to customer support.
            -Don't ask for the order number if you can find it in the invoice.
            """

            full_prompt = [{"role": "system", "content": system_prompt}] + st.session_state.messages

            # Call LLaMA-3.1
            client = InferenceClient(provider="fireworks-ai", api_key=API_KEY)
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=full_prompt
            )

            response = completion.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": response})

        except Exception as e:
            st.error(f"Error: {e}")

# --- DISPLAY MESSAGES ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
