AI Invoice Data Extractor + Chatbot (Agent)
=========================================

Project overview
----------------
This Streamlit app is an LLM-driven invoice processing system (Azure OpenAI + LangChain) that:

1. Reads PDF invoices and extracts structured invoice information as JSON.
2. Appends extracted invoice records into a universal table (`invoice_table.csv`).
3. Builds a small VectorDB index on the universal table to power a live invoice chatbot.
4. Provides a per-PDF evaluation tab to compare extraction results (ground truth from a newly uploaded PDF) with the stored universal table (prediction) and compute precision/recall/F1/accuracy.

The agent uses three tools:
- `ReadPDF` — extract raw text from uploaded PDFs (PyPDFLoader).
- `InvoiceExtractor` — call LLM to parse invoice text and return structured JSON.
- `InvoiceQuery` — index the universal table text in a Chroma vector store and answer queries using RetrievalQA.

Key features
------------
- Multi-file PDF upload and batch processing.
- LLM-based JSON extraction and validation (LLM invoked via LangChain wrapper for Azure OpenAI).
- Universal CSV table storing invoices and a download button to export all records.
- Live chat agent that can use tools to answer questions about the stored invoices.
- Evaluation tab that compares fields between a freshly uploaded PDF and a matched invoice in the universal table and computes classification metrics.

Repository structure (as implied by the single-file app)
-------------------------------------------------------
- `InvoiceDataExtractorAgent.py` — main Streamlit application (single-file app as shared).
- `invoice_table.csv` — generated universal table (created/updated at runtime).
- `invoice_chroma_index/` — persistent directory for Chroma vectorstore (created at runtime).
- `.env` — environment variables (not included in repo; add locally).
- `README.txt` — this file (explain how to use & set up locally).

Environment & dependencies
--------------------------
Minimum recommended Python environment:
- Python 3.9+ (3.10 or 3.11 recommended)

Core Python packages used by the app (install via pip):
- streamlit
- python-dotenv
- pandas
- scikit-learn
- langchain
- langchain-community (document loaders & vectorstore wrappers)
- langchain-openai (Azure OpenAI wrappers used in your code)
- chromadb (Chroma vector store)
- openai (only if you also use OpenAI client directly elsewhere)
- PyPDF2 or pypdf (as a dependency of PyPDFLoader)

Install example (use a virtualenv):

```bash
python -m venv .venv
source .venv/bin/activate   # on Windows use: .venv\Scripts\activate
pip install --upgrade pip
pip install streamlit python-dotenv pandas scikit-learn langchain langchain-community langchain-openai chromadb pypdf
```

Environment variables
---------------------
Create a `.env` file in the same folder and set the following values (names match the variables read in the code):

```
AZURE_OPENAI_ENDPOINT=https://<your-azure-openai-endpoint>
AZURE_OPENAI_DEPLOYMENT_NAME=<chat-completion-deployment-name>
AZURE_OPENAI_API_KEY=<your-azure-openai-key>
AZURE_OPENAI_API_VERSION=<api_version>         # e.g. 2023-05-15 or your configured API version
AZURE_OPENAI_MODEL_NAME=<model_name>           # e.g. gpt-4o, gpt-4, or your deployed model
AZURE_OPENAI_EMBEDDING_MODEL_NAME=<emb_model> # embedding model name
AZURE_OPENAI_EMBEDDING_MODEL_DEPLOYMENT=<embedding-deployment-name>
```

Notes:
- Make sure your Azure OpenAI resource has the models deployed and that the deployment names match the values above.
- Keep your API keys safe. Do NOT commit `.env` to source control.

How to run
----------
From the project directory run:

```bash
streamlit run InvoiceDataExtractorAgent.py
```

This will open the Streamlit UI in your browser with three tabs:
- **📄 Invoice Extractor** — upload PDFs, run extraction, view & download the universal table.
- **💬 Invoice Chatbot** — ask natural language questions about invoices stored in the universal table; powered by the agent/tooling.
- **📈 Evaluation Metrics** — upload a PDF and evaluate the extracted fields vs. the matching row in the universal table.

How the core flow works (high level)
------------------------------------
1. User uploads one or more invoice PDFs in the Extractor tab.
2. For each PDF, `ReadPDF` uses `PyPDFLoader` to extract raw text.
3. `InvoiceExtractor` sends a chat prompt to the Azure LLM which is expected to return a JSON object containing invoice fields.
4. Your app parses the returned JSON and appends it to `st.session_state["invoice_table"]` and writes `invoice_table.csv`.
5. The Chatbot tab loads `invoice_table.csv`, builds or uses a persisted Chroma index of concatenated invoice text, and uses `RetrievalQA` to answer user queries.
6. The Evaluation tab runs the same extractor on a newly uploaded PDF and compares field-by-field with the stored row (by `Invoice_ID`) and computes metrics using `sklearn`.

Important implementation notes & suggestions
------------------------------------------
- **JSON-only response requirement**: your prompt instructs the model to "Return only valid JSON." In practice you should wrap parsing in robust error handling and consider using a JSON schema checker (e.g., `jsonschema`) or a small post-LLM sanitizer to avoid crashes when the model returns non-JSON or malformed JSON.

- **Invoice_ID uniqueness**: you skip adding rows when the `Invoice_ID` already exists — good. Consider normalizing Invoice_ID types (strip whitespace, lowercase if alphanumeric) to reduce false duplicates.

- **Numeric parsing**: you attempt to coerce numeric fields (price, amount) from strings. Consider a small helper function that normalizes currency symbols, thousand separators, and also handles negative balances.

- **Chroma vector store persistence**: the code writes to `./invoice_chroma_index`. If invoices change, you may need to re-create the index or implement incremental updates.

- **Agent & tool security**: validate user-uploaded PDFs/contents if your app runs in a shared environment. Avoid logging secrets or raw LLM outputs that might contain PII.

- **LLM invocation**: you call `llm.invoke(message)` — depending on the LangChain wrapper versions you use, the official patterns may be different; ensure your installed `langchain-openai` package supports this method or replace with the LangChain `llm.generate()` / `llm()` usual calls.

- **Concurrency & session state**: Streamlit can re-run the script on interactions. You already guard the universal table in `st.session_state`, but make sure to handle concurrent users carefully if deployed multi-user (use a server-side DB instead of a local CSV).

- **Unit tests & evaluation**: Right now the evaluation uses a per-field match (binary equality) and treats ground-truth as always 1. This gives basic metrics but consider field-level fuzzy matching (e.g., numeric tolerance, string similarity) and more granular item-level checks for `Items` lists.

Troubleshooting & common errors
-------------------------------
- **Malformed JSON from LLM**: wrap `json.loads()` in try/except and provide a fallback (show raw LLM output in the UI and allow manual edit/save).
- **Missing Env Vars**: Streamlit/LLM may fail if environment variables are unset — check `.env` and use `st.error()` to show missing variables.
- **Chroma import issues**: Ensure `chromadb` and `langchain-community` are compatible with your LangChain version.
- **PyPDFLoader errors**: If you see PDF parsing issues, try switching the loader (pypdf directly) or pre-transforming PDFs (OCR if scanned images).

Security & privacy considerations
--------------------------------
- Invoices often contain PII and financial data. Ensure your Azure/OpenAI keys are secure and the app is run in a compliant environment.
- If deploying publicly, consider disabling file uploads or adding authentication and data retention policies.

Extending the project
---------------------
- Add an explicit schema validation step and a UI to correct extraction errors.
- Store `invoice_table` in a real database (Postgres, Snowflake, Databricks) for multi-user, scale, and ACID guarantees.
- Add a background worker (e.g., Celery or RQ) to process PDF uploads asynchronously (note: the current system processes synchronously in the request flow).
- Build an admin UI to re-index the vectorstore after bulk updates or deletions.

License & attribution
---------------------
Add a license file suitable to your project (MIT, Apache-2.0, etc.) if you plan to publish.

Contact / Author
----------------
This README was generated to describe and document the provided Streamlit app `InvoiceDataExtractorAgent.py`. For feature requests or bug reports, please update the README or add an issues/CONTRIBUTING section.



