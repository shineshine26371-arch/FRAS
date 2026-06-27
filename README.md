# File & Report Automation System (FRAS)

A local web app that lets you upload documents, automatically extracts and summarizes their content using OpenRouter (powered by Google Gemini), stores everything for later search, and generates structured reports on demand.

## Tech Stack

- **UI:** Streamlit
- **AI Processing:** OpenRouter API (using google/gemini-2.5-flash)
- **Storage:** Local filesystem + SQLite
- **Language:** Python 3.11+

## Setup

1. **Clone or download this repository.**

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   # or
   source .venv/bin/activate   # macOS/Linux
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Get an OpenRouter API key:**
   - Sign up at [openrouter.ai](https://openrouter.ai/)
   - Go to [Keys](https://openrouter.ai/keys) and create a new API key

5. **Configure your environment:**
   - Copy `.env.example` to `.env`
   - Replace `your_openrouter_api_key_here` with your actual OpenRouter API key
   ```bash
   cp .env.example .env
   # Then edit .env with your key
   ```

   Your `.env` should look like:
   ```env
   OPENROUTER_API_KEY=sk-or-v1-...
   
   # Optional: App attribution for OpenRouter leaderboards
   # OPENROUTER_HTTP_REFERER=https://your-site.com
   # OPENROUTER_APP_TITLE=Your App Name
   ```

   > **Note:** The optional `OPENROUTER_HTTP_REFERER` and `OPENROUTER_APP_TITLE` headers allow your app to appear on OpenRouter leaderboards. See [App Attribution](https://openrouter.ai/docs/app-attribution) for details.

6. **Run the app:**
   ```bash
   streamlit run fras.py
   ```

7. **Open your browser** to the URL shown in the terminal (usually `http://localhost:8501`).

## Usage

### Upload Tab
- Upload PDF, DOCX, TXT, PNG, or JPG files
- Click "Upload & Process" to save files and run AI extraction
- Progress is shown for each file in the batch
- Errors are displayed persistently (not flashing) if processing fails

### Library Tab
- View all uploaded documents in a searchable table
- Search across filenames, summaries, and key points
- Click "View Details" to see full extracted content and download the original file

### Reports Tab
- Multi-select processed documents
- Click "Generate Report" to create a structured Markdown report
- Download as `.md` or `.docx`

## Project Structure

```
.
├── fras.py             # Streamlit entrypoint (tabs: Upload, Library, Reports)
├── db.py               # SQLite schema + CRUD helpers
├── openrouter_client.py # OpenRouter API wrapper with rate limiting + retry
├── requirements.txt    # Python dependencies
├── .env.example        # Example environment config
├── .env                # Your actual API key (not committed)
├── fras.db             # SQLite database (created on first run)
└── storage/
    └── files/          # Uploaded raw files
```

## Notes

- **Rate limiting:** The app enforces a minimum 4-second delay between API calls to respect rate limits. Batch uploads are processed sequentially.
- **Error handling:** Failed extractions are logged in the UI and marked with `status = 'failed'` in the database. Errors persist across reruns.
- **No hardcoded secrets:** The API key is loaded from `.env` via `python-dotenv`.
- **OpenRouter:** Uses the `google/gemini-2.5-flash` model via OpenRouter. Get your API key at https://openrouter.ai/keys

## Roles & Permissions (MVP)

The app includes a simple role selector in the sidebar (no real authentication):

- **Owner:** Full access — can upload files, delete documents, retry failed processing, and download original files.
- **Viewer:** Read-only access — can browse the library, search, view document details, and generate reports. Upload, delete, retry, and download actions are hidden.

> **Note:** This is an MVP simplification. In a production system, this would be replaced with proper authentication and role-based access control.

## Categories

Documents can be tagged with categories for better organization:

- Categories are extracted automatically by Gemini during processing (e.g., "Report", "Invoice", "Memo").
- You can filter the Library view by category using the dropdown filter.
- Categories are stored in the database and can be used to organize large document collections.

## Out of Scope (MVP)

- User accounts / authentication
- Cloud deployment / multi-user concurrency
- Vector embeddings or semantic search
- File editing or versioning
