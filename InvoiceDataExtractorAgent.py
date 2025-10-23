import os
from dotenv import load_dotenv, find_dotenv
import streamlit as st
import json
import tempfile
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import Tool, AgentExecutor, create_tool_calling_agent
from langchain.chains import RetrievalQA


# ================================================================
# LOAD ENVIRONMENT VARIABLES Step-1
# ================================================================
load_dotenv(find_dotenv())

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_MODEL_NAME = os.getenv("AZURE_OPENAI_MODEL_NAME")
AZURE_OPENAI_EMBEDDING_MODEL_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL_NAME")
AZURE_OPENAI_EMBEDDING_MODEL_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL_DEPLOYMENT")



# ================================================================
# INITIALIZE LLM AND EMBEDDINGS Step-2
# ================================================================
llm = AzureChatOpenAI(
    deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
    openai_api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
    model_name=AZURE_OPENAI_MODEL_NAME,
    temperature=0
)

embeddings = AzureOpenAIEmbeddings(
    deployment=AZURE_OPENAI_EMBEDDING_MODEL_DEPLOYMENT,
    model=AZURE_OPENAI_EMBEDDING_MODEL_NAME,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY
)




# ================================================================
# STREAMLIT LAYOUT Step-3 
# ================================================================
st.set_page_config(page_title="📄 Invoice Extractor + Chatbot (Agent)", layout="wide")
tab1, tab2,tab3 = st.tabs(["📄 Invoice Extractor", "💬 Invoice Chatbot", "📈 Evaluation Metrics"])





# ================================================================
# TOOL 1: READ RAW TEXT FROM UPLOADED PDF for multiple uploads Step-4
# ================================================================
def read_pdf_content(file_bytes: bytes) -> str:
    """Reads PDF bytes and extracts text."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_file.flush()
        loader = PyPDFLoader(temp_file.name)
        pages = loader.load()
        text = "\n".join([p.page_content for p in pages])
    return text

read_pdf_tool = Tool(
    name="ReadPDF",
    func=read_pdf_content,
    description="Reads and returns text content from a PDF file."
)





# ================================================================
# TOOL 2: EXTRACT INVOICE DATA FROM Text Step-5
# ================================================================
def extract_invoice_from_text(text: str) -> str:
    template = """
    You are an AI that extracts structured invoice data.
    Extract the following fields in JSON format:

    - Customer Name
    - Ship To 
    - Vendor
    - Invoice_ID
    - Date (YYYY-MM-DD)
    - Ship Mode
    - Balance Due
    - Amount
    - Discount (enter 0 if not specified)
    - Shipping
    - Currency
    - Items: For each item, include:
        - Item Name
        - Description
        - Category
        - Item Code
        - Quantity
        - Unit_Price
        - Total
    - Order ID

    *Return only valid JSON.*

    Invoice text:
    {input_text}
    """
    prompt = ChatPromptTemplate.from_template(template)
    message = prompt.format_messages(input_text=text)
    response = llm.invoke(message)
    return response.content.strip()

invoice_extractor_tool = Tool(
    name="InvoiceExtractor",
    func=extract_invoice_from_text,
    description="Extracts structured JSON data from invoice text."
)




# ================================================================
# TOOL 3: QUERY EXISTING INVOICE DATA Step-6
# ================================================================
def query_invoices(query: str, df_path="invoice_table.csv") -> str:
    if not os.path.exists(df_path):
        return "No invoice data found. Please process invoices first."

    df = pd.read_csv(df_path)
    invoice_texts = "\n".join([
        ", ".join([f"{col}: {row[col]}" for col in df.columns])
        for _, row in df.iterrows()
    ])

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(invoice_texts)

    vector_dir = "./invoice_chroma_index"
    os.makedirs(vector_dir, exist_ok=True)

    vectorstore = Chroma.from_texts(chunks, embedding=embeddings, persist_directory=vector_dir)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    result = qa_chain.invoke({"query": query})
    return result["result"]

invoice_query_tool = Tool(
    name="InvoiceQuery",
    func=query_invoices,
    description="Answers questions about invoices stored in the universal table."
)








# ================================================================
# AGENT SETUP Step-7
# ================================================================
tools = [read_pdf_tool, invoice_extractor_tool, invoice_query_tool]

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an intelligent invoice assistant that can read, extract, and analyze invoices using the provided tools."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# ✅ use the modern tool-calling agent (GPT-5 compatible)
agent = create_tool_calling_agent(llm, tools, prompt_template)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=ConversationBufferWindowMemory()
)




# ================================================================
# TAB 1: INVOICE EXTRACTION
# ================================================================
with tab1:
    st.title("📄 AI Invoice Data Extractor (Azure OpenAI + LangChain Agent)")
    uploaded_files = st.file_uploader("Upload your Invoice PDFs", type=["pdf"], accept_multiple_files=True)

    # ---------------------------
    # Initialize session table 
    # ---------------------------
    if "invoice_table" not in st.session_state:
        if os.path.exists("invoice_table.csv"):
            st.session_state["invoice_table"] = pd.read_csv("invoice_table.csv")
        else:
            st.session_state["invoice_table"] = pd.DataFrame(columns=[
                "Customer Name","Ship To","Vendor","Invoice_ID","Date","Ship Mode",
                "Balance Due","Amount","Discount","Shipping","Currency","Items","Order ID"
            ])

    # ---------------------------
    # Process uploaded files
    # ---------------------------
    if uploaded_files and st.button("🚀 Process Invoices"):
        for uploaded_file in uploaded_files:
            st.write(f"📄 Processing *{uploaded_file.name}*...")
            pdf_bytes = uploaded_file.getvalue()
            pdf_text = read_pdf_content(pdf_bytes)

            with st.spinner(f"🤖 Extracting data from {uploaded_file.name}..."):
                json_output = extract_invoice_from_text(pdf_text)

            try:
                parsed_json = json.loads(json_output)
                formatted_json = json.dumps(parsed_json, indent=2)
                st.success(f"✅ Extraction successful for *{uploaded_file.name}*")
                st.json(parsed_json)

                row_data = parsed_json.copy()
                row_data["Items"] = json.dumps(parsed_json.get("Items", []))
                invoice_id = parsed_json.get("Invoice_ID")

                if not invoice_id:
                    st.warning(f"⚠️ No Invoice_ID found for {uploaded_file.name}. Skipping.")
                    continue

                existing_invoices = st.session_state["invoice_table"]
                if not existing_invoices.empty and "Invoice_ID" in existing_invoices.columns:
                    if invoice_id in existing_invoices["Invoice_ID"].astype(str).values:
                        st.info(f"ℹ️ Invoice {invoice_id} already exists. Skipping duplicate.")
                        continue

                st.session_state["invoice_table"] = pd.concat(
                    [existing_invoices, pd.DataFrame([row_data])],
                    ignore_index=True
                )
                st.success(f"✅ Invoice {invoice_id} added to the universal table!")

                st.session_state["invoice_table"].to_csv("invoice_table.csv", index=False)

                st.download_button(
                    label=f"💾 Download JSON for {uploaded_file.name}",
                    data=formatted_json,
                    file_name=f"{uploaded_file.name.split('.')[0]}_invoice.json",
                    mime="application/json"
                )

            except Exception:
                st.error(f"❌ Failed to parse JSON for {uploaded_file.name}. Showing raw output:")
                st.code(json_output, language="json")

    # ---------------------------
    # Reset button (always visible)
    # ---------------------------
    if st.button("🗑️ Reset Universal Table"):
        try:
            # Delete CSV if exists
            if os.path.exists("invoice_table.csv"):
                os.remove("invoice_table.csv")

            # Reset the session DataFrame to empty
            st.session_state["invoice_table"] = pd.DataFrame(columns=[
                "Customer Name","Ship To","Vendor","Invoice_ID","Date","Ship Mode",
                "Balance Due","Amount","Discount","Shipping","Currency","Items","Order ID"
            ])

            # Optional: clear chat history too
            if "chat_history" in st.session_state:
                st.session_state["chat_history"] = []

            st.success("✅ Universal table reset successfully!")

        except Exception as e:
            st.error(f"❌ Error resetting table: {e}")

    # ---------------------------
    # Display universal table (always visible)
    # ---------------------------
    st.subheader("📊 Universal Invoice Table")
    st.dataframe(st.session_state["invoice_table"], use_container_width=True)

    st.download_button(
        label="💾 Download All Invoices (CSV)",
        data=st.session_state["invoice_table"].to_csv(index=False).encode("utf-8"),
        file_name="universal_invoice_table.csv",
        mime="text/csv"
    )

    # ---------------------------
    # Info if no uploaded files
    # ---------------------------
    if not uploaded_files:
        st.info("📥 Please upload one or more PDF files to begin.")





# ================================================================
# TAB 2: CHATBOT (LIVE AGENT)
# ================================================================
with tab2:
    st.title("💬 Invoice Chatbot")

    if not os.path.exists("invoice_table.csv"):
        st.warning("⚠️ Please process and save invoices first in the 'Invoice Extractor' tab.")
    else:
        st.success("✅ Chatbot ready! Ask your questions below.")

        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        user_query = st.text_input("Ask a question about invoices or documents:")

        if user_query:
            with st.spinner("🤖 Agent Thinking..."):
                response = agent_executor.invoke({"input": user_query})
                answer = response["output"]

            st.session_state["chat_history"].append({"question": user_query, "answer": answer})

            st.markdown("### 🧠 Agent’s Answer:")
            st.write(answer)

            with st.expander("💬 Conversation History"):
                for i, msg in enumerate(st.session_state["chat_history"], 1):
                    st.markdown(f"*Question {i}:* {msg['question']}")
                    st.markdown(f"*Answer {i}:* {msg['answer']}")
                    st.markdown("---")




# ================================================================
# TAB 3: EVALUATION METRICS (PER INVOICE PDF)
# ================================================================

with tab3:
    st.title("📊 Invoice Evaluation by PDF")

    # Load universal table
    if "invoice_table" not in st.session_state or st.session_state["invoice_table"].empty:
        st.warning("⚠️ No extracted invoices found. Please process invoices first.")
    else:
        st.markdown("Upload a PDF invoice to evaluate against the universal table")
        eval_pdf = st.file_uploader("Upload PDF for Evaluation", type=["pdf"], key="eval_pdf_upload")

        if eval_pdf is not None:
            pdf_bytes = eval_pdf.getvalue()

            with st.spinner("🤖 Extracting invoice data for evaluation..."):
                try:
                    pdf_text = read_pdf_content(eval_pdf.getvalue())
                    extracted_json = extract_invoice_from_text(pdf_text)
                    extracted_data = json.loads(extracted_json)

                    invoice_id = extracted_data.get("Invoice_ID")
                    if not invoice_id:
                        st.error("⚠️ Invoice_ID not found in the uploaded PDF.")
                    else:
                        # Match against universal table (prediction)
                        universal_df = st.session_state["invoice_table"]
                        match = universal_df[universal_df["Invoice_ID"].astype(str) == str(invoice_id)]

                        if match.empty:
                            st.error(f"⚠️ Invoice_ID {invoice_id} not found in universal table.")
                        else:
                            pred_row = match.iloc[0].to_dict()

                            # Compare fields (PDF = ground truth, Universal table = prediction)
                            fields = [c for c in pred_row.keys() if c != "Invoice_ID"]
                            y_true = []
                            y_pred = []
                            results = []

                            for field in fields:
                                gt_val = extracted_data.get(field, "")
                                pred_val = pred_row.get(field, "")

                                # Handle numeric comparison
                                try:
                                    gt_val_num = float(str(gt_val).replace(",", "").replace("$", ""))
                                    pred_val_num = float(str(pred_val).replace(",", "").replace("$", ""))
                                    match_val = int(gt_val_num == pred_val_num)
                                except:
                                    match_val = int(str(gt_val).strip() == str(pred_val).strip())

                                y_true.append(1)  # ground truth is always 1
                                y_pred.append(match_val)
                                results.append({
                                    "Field": field,
                                    "Ground Truth": gt_val,
                                    "Predicted": pred_val,
                                    "Match": match_val
                                })

                            # Convert all columns to string to avoid PyArrow errors
                            results_df = pd.DataFrame(results).astype(str)

                            st.subheader(f"✅ Field-by-field Match for Invoice_ID: {invoice_id}")
                            st.dataframe(results_df, use_container_width=True)

                            # Compute metrics
                            from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

                            precision = precision_score(y_true, y_pred, zero_division=0)
                            recall = recall_score(y_true, y_pred, zero_division=0)
                            f1 = f1_score(y_true, y_pred, zero_division=0)
                            accuracy = accuracy_score(y_true, y_pred)

                            st.subheader("📌 Evaluation Metrics")
                            st.write(f"**Accuracy:** {accuracy:.3f}")
                            st.write(f"**Precision:** {precision:.3f}")
                            st.write(f"**Recall:** {recall:.3f}")
                            st.write(f"**F1-score:** {f1:.3f}")

                except Exception as e:
                    st.error(f"❌ Error extracting/evaluating invoice: {e}")
        else:
            st.info("📥 Please upload a PDF to evaluate.")



#   streamlit run "c:\Users\devas\OneDrive\Documents\Coding\Generative and Agentic AI\Capstone Project\InvoiceDataExtractorAgent.py"