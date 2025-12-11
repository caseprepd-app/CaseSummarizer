# Project Specification: LocalScribe (v2.0 - FINAL)

## 1. Core Mission
Build a **100% offline, private, commercial Windows desktop application** ("LocalScribe") for court reporters. The app's primary function is to solve the PII/PHI liability problem by ensuring sensitive legal documents *never* leave the user's computer.

**Key Principle:** This app processes MULTIPLE documents (complaint, answer, exhibits, motions, etc.) to generate ONE comprehensive case-level summary, not individual document summaries.

---

## 2. Core Features (The 5-Step Pipeline)

> **Strategic Pivot (Session 29):** The application has evolved from summarization-first to **Q&A-first**. Court reporters need quick case familiarization—Q&A results serve as handoff documents for colleagues. Summarization remains available but is off by default due to 30+ minute processing times.

The app executes a 5-step process:

1.  **File Ingest:** User selects one or more documents (PDF, TXT, RTF).
2.  **Pre-processing (The "Sniffer" & "Cleaner"):** The app extracts clean, usable text from each document and provides OCR confidence scores.
3.  **File Selection:** User reviews OCR confidence scores and selects which documents to include in processing.
4.  **AI Processing (The "Brain"):** Selected documents are indexed in a FAISS vector store for Q&A, and optionally fed to Ollama for summarization.
5.  **Output Generation:** The app displays three outputs:
    - **Names & Vocabulary (Primary):** Combined people + technical terms extracted via NER + LLM reconciliation. Shows "Found By" column indicating whether each item was found by Both, NER only, or LLM only. Progressive output: NER results shown immediately, enhanced with LLM results when complete. (Session 45)
    - **Q&A Results:** Document Q&A using hybrid retrieval (BM25+ + FAISS) with LlamaIndex query transformation for vague questions (Session 44)
    - **Case Summary (Optional):** AI-generated synthesis via Ollama (off by default)

> **Session 45 Architecture Update:** The extraction pipeline now uses unified semantic chunking with token enforcement (tiktoken). Single chunking pass serves both LLM extraction and Q&A indexing. Key changes:
> - Combined extraction prompt extracts both people (with roles) and vocabulary in one LLM call per chunk
> - Simplified UI checkboxes: "Extract Names & Vocabulary" (ON), "Enable Q&A" (ON), "Generate Summary" (OFF)
> - Output pane has distinct background color with progress badge showing data source (NER only → NER + LLM)

---

## 3. Technical Stack

### 3.1 Required Technologies
* **Language:** Python 3.10+
*   **Virtual Environment:** `.venv` (mandatory for all development; always use `.\\.venv\Scripts\python.exe` for any Python-related commands to ensure isolation and prevent global installations).
* **UI Framework:** **CustomTkinter** (Modern and robust Tkinter UI library)
* **Local AI Engine:** **`llama-cpp-python`** (to run GGUF models on CPU with streaming support)
* **Local OCR Engine:** **Tesseract** (via `pytesseract`). Must be bundled with the application.
* **PDF Handling:** **`pdfplumber`** (MIT license - safe for commercial use)
* **PDF to Image:** **`pdf2image`** (for OCR preprocessing)
* **NLP Tools:** **`nltk`** for word frequency lists and basic text processing
* **Query Transformation:** **LlamaIndex** (`llama-index-core`, `llama-index-llms-ollama`) for expanding vague Q&A queries into specific search terms (Session 44)
* **Token Counting:** **`tiktoken`** for accurate token counting in semantic chunking (Session 45)
* **Packaging:** **PyInstaller** (to create standalone `.exe`)

### 3.2 External Data Dependencies
* **Google Word Frequency List:** For rare word detection (333k most common words)
* **Downloadable Filter Lists (from Dropbox):**
  * Legal keyword lists (e.g., `legal_keywords_ny.txt`, `legal_keywords_california.txt`)
  * Document type priority rules (e.g., `document_priority_ny.txt`)
* **AI Models:** Gemma 2 GGUF models (hosted on Dropbox, downloaded via license system)

### 3.3 License Compliance
* **Tesseract:** Apache 2.0 ✓
* **pdfplumber:** MIT ✓
* **Gemma 2 Models:** Google's terms allow commercial use with attribution ✓
* **Attribution requirement:** Display in About dialog: "Powered by Google Gemma 2 models"

---

## 4. Model Distribution & Licensing System

### 4.1 Model Hosting Strategy
* **Host models on Dropbox Business** account (1 TB bandwidth/day limit)
* Two model sizes available:
  * **Standard Model:** `gemma-2-9b-it-q4_k_m.gguf` (~7GB) - Fast, good quality
  * **Pro Model:** `gemma-2-27b-it-q4_k_m.gguf` (~22GB) - Slow, best quality
* Bandwidth capacity: ~143 downloads/day of Standard model within 1TB limit

### 4.2 License Server Integration

**Note:** The license server itself is a separate system (outside this spec). LocalScribe communicates with it via HTTP API.

**License System Flow:**
1.  **User Registration:** User obtains a license key through the developer's website/payment system
2.  **App Launch:** On first launch, LocalScribe prompts for license key
3.  **License Validation API Call:**
    ```
    POST /api/validate_license
    Body: {"license_key": "USER_KEY_HERE", "action": "download_request"}
    Response: {
      "valid": true/false,
      "download_allowed": true/false,
      "download_url": "https://dropbox.com/...",
      "quota_remaining": {"daily": 0, "weekly": 1, "monthly": 3, "annual": 10},
      "next_available": "2025-11-13T00:00:00Z"
    }
    ```
4.  **Download Quotas per License:**
    * Daily: 1 download
    * Weekly: 2 downloads
    * Monthly: 4 downloads
    * Annual: 12 downloads
5.  **Quota Exceeded Behavior:** Display message: "Download limit reached. Your license allows [X] downloads per [period]. Next download available on [date]."
6.  **Download Process:** If approved, use the returned Dropbox URL to download model with progress bar

**Implementation Notes:**
* Store license key locally (encrypted): `%APPDATA%/LocalScribe/license.dat`
* Cache license validation for 24 hours to avoid constant API calls
* Model storage: `%APPDATA%/LocalScribe/models/`
* Check for model updates weekly (compare file checksums via API)

### 4.3 Bandwidth Suspension Handling
* **Dropbox behavior:** If bandwidth limit exceeded, shared links are temporarily suspended for 24-72 hours (first offense), longer for subsequent offenses
* **User message if download fails:** "Model download temporarily unavailable. Please try again in 24 hours or contact support."
* **Mitigation:** License server tracks downloads and can rotate between multiple Dropbox accounts if needed

---

## 5. Pre-processing Pipeline (The "Cleaner")

### 5.1 Overview
The extractor is the most critical component. It processes each document independently to extract clean, readable text and assign confidence scores.

**Input:** List of file paths (PDF, TXT, RTF)
**Output:** For each file:
* Clean text string
* OCR confidence score (0-100%)
* Processing status (success/warning/error)

### 5.2 Processing Flow (Per Document)

#### Step 1: File Type Detection
```python
if file.endswith('.txt') or file.endswith('.rtf'):
    text = read_text_file(file)
    confidence = 100
    method = "direct_read"
elif file.endswith('.pdf'):
    # Proceed to Step 2
```

#### Step 2: PDF Text Extraction
```python
import pdfplumber

text = ""
with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        text += page.extract_text() or ""
```

#### Step 3: Heuristic Check (Digital vs Scanned)
```python
# Load NLTK English words corpus
from nltk.corpus import words
english_words = set(words.words())

# Tokenize and check
tokens = text.lower().split()
valid_words = sum(1 for token in tokens if token in english_words)
total_words = len(tokens)

if total_words > 0:
    dictionary_percentage = (valid_words / total_words) * 100
else:
    dictionary_percentage = 0

# Decision logic
if dictionary_percentage > 60 and len(text) > 1000:
    confidence = 100
    method = "digital_text"
    # Skip to Step 5 (cleaning)
else:
    # Proceed to Step 4 (OCR)
```

#### Step 4: OCR Processing (If Needed)
```python
from pdf2image import convert_from_path
import pytesseract

# Convert PDF to images
images = convert_from_path(pdf_path, dpi=300)

# OCR each page
ocr_text = ""
for image in images:
    ocr_text += pytesseract.image_to_string(image) + "\n"

# Calculate confidence using same dictionary check
tokens = ocr_text.lower().split()
valid_words = sum(1 for token in tokens if token in english_words)
confidence = (valid_words / len(tokens)) * 100 if tokens else 0

text = ocr_text
method = "ocr"
```

#### Step 5: Text Cleaning Rules

**Load Dynamic Keyword Lists:**
```python
# Download and cache from Dropbox (check weekly for updates)
legal_keywords = load_keywords_list("legal_keywords_{jurisdiction}.txt")
# Example content: ["COURT", "PLAINTIFF", "DEFENDANT", "APPEARANCES", "SUPREME"]
```

**Rule 1: Line Filtering**
```python
cleaned_lines = []
for line in text.split('\n'):
    # Keep line if it passes ALL these tests:
    if len(line) > 15:  # Minimum length
        has_lowercase = any(c.islower() for c in line)
        is_legal_header = (line.isupper() and 
                          len(line) < 50 and 
                          any(keyword in line for keyword in legal_keywords))
        
        # Count character types
        alpha_count = sum(c.isalpha() for c in line)
        other_count = sum(not c.isalpha() and not c.isspace() for c in line)
        
        if (has_lowercase or is_legal_header) and alpha_count > other_count:
            cleaned_lines.append(line)
```

**Rule 2: De-hyphenation**
```python
# Rejoin words split by line breaks (e.g., "defen-\ndant" → "defendant")
text = '\n'.join(cleaned_lines)
text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)

# Preserve legitimate hyphenated compounds (e.g., "attorney-client")
# This is handled by only removing hyphen+newline, not hyphen+space
```

**Rule 3: Whitespace Normalization**
```python
# Remove excess blank lines (keep max 1 between paragraphs)
text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
text = text.strip()
```

### 5.3 Error Handling During Pre-processing

**Zero-length cleaned text:**
```python
if len(cleaned_text.strip()) == 0:
    status = "error"
    message = "Unable to extract readable text. File may be corrupted or contain only images."
    # Mark file for skipping with warning dialog
```

**File size limits:**
```python
file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

if file_size_mb > 500:
    status = "error"
    message = "File exceeds maximum size (500MB). Please split file or extract relevant sections."
    # Reject file
elif file_size_mb > 100:
    # Show warning dialog: "Large file detected (XXX MB). Processing may take significantly longer. Continue?"
    # If user clicks No, skip file
```

**Password-protected PDFs:**
```python
try:
    with pdfplumber.open(pdf_path) as pdf:
        # Attempt to access
except Exception as e:
    if "password" in str(e).lower() or "encrypted" in str(e).lower():
        status = "error"
        message = "PDF is password-protected and cannot be processed."
```

### 5.4 Multi-File Summary & User Selection Interface

After processing all files, display a **File Review Table**:

| Filename | Status | Method | OCR Confidence | Pages | Size | Include |
|----------|--------|--------|----------------|-------|------|---------|
| complaint.pdf | ✓ Ready | Digital | 100% | 45 | 2.3 MB | ☑ |
| exhibit_a.pdf | ⚠ Low Quality | OCR | 60% | 12 | 856 KB | ☐ |
| exhibit_b.pdf | ⚠ Warning | OCR | 75% | 8 | 1.1 MB | ☑ |
| deposition.pdf | ✓ Ready | OCR | 89% | 120 | 8.7 MB | ☑ |
| motion.pdf | ✓ Ready | Digital | 100% | 23 | 1.8 MB | ☑ |
| corrupted.pdf | ✗ Failed | Error | 0% | - | - | - |

**UI Elements:**
* **Warning banner:** "⚠ Files with confidence <70% may produce unreliable results"
* **Checkboxes:** Pre-checked for files with confidence ≥70%, unchecked for <70%
* **Status icons:** ✓ (green), ⚠ (yellow), ✗ (red)
* **Buttons:**
  * "Select All" / "Deselect All"
  * "Process Selected Files" (shows estimated time: "~12 minutes with Standard model")
  * "Cancel" (return to file selection)

**Failed files behavior:**
* Show warning dialog listing failed files: "The following files could not be processed: [list]. Continue with remaining files?"
* Remove failed files from processing queue
* Log errors to `%APPDATA%/LocalScribe/logs/processing.log`

---

## 6. Document Prioritization System

### 6.1 Purpose
When combined document text exceeds the AI model's context window (~6000 tokens for safe processing), the app must intelligently truncate less important documents while preserving critical case documents in full.

### 6.2 User Settings (Set Once, Use Always)

**Settings Menu → Document Priority Rules**

Users configure document priority ONCE based on their workflow. This becomes their default for all future cases.

**Configuration Interface:**
```
┌─────────────────────────────────────────────┐
│ Document Priority Rules                      │
│                                              │
│ Jurisdiction Preset: [New York State ▼]     │
│                                              │
│ HIGH PRIORITY (never truncated):             │
│ ┌──────────────────────────────────────────┐│
│ │ • complaint                               ││
│ │ • answer                                  ││
│ │ • bill of particulars                     ││
│ │ • summons                                 ││
│ └──────────────────────────────────────────┘│
│                                              │
│ MEDIUM PRIORITY (truncate proportionally):   │
│ ┌──────────────────────────────────────────┐│
│ │ • motion                                  ││
│ │ • affidavit                               ││
│ │ • exhibit                                 ││
│ └──────────────────────────────────────────┘│
│                                              │
│ LOW PRIORITY (truncate first):               │
│ ┌──────────────────────────────────────────┐│
│ │ • notice                                  ││
│ │ • certificate                             ││
│ │ • cover                                   ││
│ │ • stipulation                             ││
│ └──────────────────────────────────────────┘│
│                                              │
│ [Load Preset ▼] [Save as Custom] [Reset]    │
└─────────────────────────────────────────────┘
```

**Preset Options:**
* New York State (default shown above)
* California
* Federal Court
* Custom (user-defined)

**Settings Storage:**
```json
// %APPDATA%/LocalScribe/config/priority_rules.json
{
  "jurisdiction": "new_york",
  "high_priority_keywords": ["complaint", "answer", "bill of particulars", "summons"],
  "medium_priority_keywords": ["motion", "affidavit", "exhibit", "deposition"],
  "low_priority_keywords": ["notice", "certificate", "cover", "stipulation"]
}
```

### 6.3 Auto-Categorization at Runtime

**When user selects files for processing:**
```python
def categorize_document(filename, priority_rules):
    filename_lower = filename.lower()
    
    # Check high priority keywords
    for keyword in priority_rules['high_priority_keywords']:
        if keyword in filename_lower:
            return "HIGH"
    
    # Check medium priority
    for keyword in priority_rules['medium_priority_keywords']:
        if keyword in filename_lower:
            return "MEDIUM"
    
    # Check low priority
    for keyword in priority_rules['low_priority_keywords']:
        if keyword in filename_lower:
            return "LOW"
    
    # Default to MEDIUM if no match
    return "MEDIUM"
```

**Display in File Review Table:**
Add a "Priority" column showing HIGH/MEDIUM/LOW auto-assignment based on filename.

### 6.4 Truncation Logic (When Context Window Exceeded)

```python
def prepare_combined_text(selected_files, priority_rules, max_tokens=6000):
    """
    Combine text from multiple documents, truncating intelligently if needed.
    """
    # Categorize files
    high_priority = [f for f in selected_files if f.priority == "HIGH"]
    medium_priority = [f for f in selected_files if f.priority == "MEDIUM"]
    low_priority = [f for f in selected_files if f.priority == "LOW"]
    
    # Estimate tokens (rough: 1 token ≈ 0.75 words)
    def estimate_tokens(text):
        return int(len(text.split()) / 0.75)
    
    combined_text = ""
    total_tokens = 0
    truncation_occurred = False
    
    # Add HIGH priority documents in full (never truncate)
    for doc in high_priority:
        separator = f"\n\n--- DOCUMENT: {doc.filename} ---\n\n"
        combined_text += separator + doc.clean_text
        total_tokens += estimate_tokens(separator + doc.clean_text)
    
    # Calculate remaining budget
    remaining_tokens = max_tokens - total_tokens
    
    if remaining_tokens <= 0:
        # Even high priority docs exceed limit - truncate proportionally
        # (This should be rare with typical case documents)
        truncation_occurred = True
        combined_text = truncate_proportionally(high_priority, max_tokens)
        return combined_text, truncation_occurred
    
    # Add MEDIUM priority (truncate proportionally if needed)
    medium_text = ""
    for doc in medium_priority:
        separator = f"\n\n--- DOCUMENT: {doc.filename} ---\n\n"
        medium_text += separator + doc.clean_text
    
    medium_tokens = estimate_tokens(medium_text)
    
    if medium_tokens <= remaining_tokens:
        combined_text += medium_text
        remaining_tokens -= medium_tokens
    else:
        # Truncate medium priority proportionally
        truncation_occurred = True
        truncation_ratio = remaining_tokens / medium_tokens
        for doc in medium_priority:
            separator = f"\n\n--- DOCUMENT: {doc.filename} ---\n\n"
            truncated = doc.clean_text[:int(len(doc.clean_text) * truncation_ratio)]
            combined_text += separator + truncated + "\n[...truncated...]"
        remaining_tokens = 0
    
    # Add LOW priority only if budget remains
    if remaining_tokens > 0:
        for doc in low_priority:
            separator = f"\n\n--- DOCUMENT: {doc.filename} ---\n\n"
            doc_text = separator + doc.clean_text
            doc_tokens = estimate_tokens(doc_text)
            
            if doc_tokens <= remaining_tokens:
                combined_text += doc_text
                remaining_tokens -= doc_tokens
            else:
                # Truncate this low priority doc
                truncation_occurred = True
                truncation_ratio = remaining_tokens / doc_tokens
                truncated = doc.clean_text[:int(len(doc.clean_text) * truncation_ratio)]
                combined_text += separator + truncated + "\n[...truncated...]"
                break
    
    return combined_text, truncation_occurred

# Display warning if truncation occurred
if truncation_occurred:
    show_warning("⚠ Some documents truncated to fit processing limits. High priority documents (complaint, answer, bill of particulars) were preserved in full.")
```

---

## 7. AI Processing (The "Brain")

### 7.1 Model Management

**Model Locations:**
* Standard Model: `%APPDATA%/LocalScribe/models/gemma-2-9b-it-q4_k_m.gguf`
* Pro Model: `%APPDATA%/LocalScribe/models/gemma-2-27b-it-q4_k_m.gguf`

**Model Loading:**
```python
from llama_cpp import Llama

def load_model(model_path, n_ctx=8192):
    """
    Load GGUF model with llama-cpp-python
    n_ctx: Context window size in tokens
    """
    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=4,  # Adjust based on CPU cores
        n_gpu_layers=0,  # CPU-only (set to >0 if GPU available)
        verbose=False
    )
    return llm
```

### 7.2 User-Controlled Settings

**Model Selection:**
* Dropdown in UI: "Standard (9B - Fast)" or "Pro (27B - Best Quality)"
* Pro model shows warning: "⚠ Pro mode takes 30-60 minutes per processing job. Best for overnight batch processing."

**Summary Length:**
* Slider: 100-500 words (default: 200)
* Real-time estimate updates as slider moves:
  * 100 words + Standard: "~5 min"
  * 200 words + Standard: "~8 min"
  * 500 words + Standard: "~15 min"
  * 200 words + Pro: "~35 min"
* Tooltip: "Longer summaries take more time to generate"

### 7.3 User Tips (Display Before Processing)

Show these tips in a collapsible panel:
```
💡 Tips for Best Results:
• Process only the most relevant documents to save time
• Extract key sections from large documents rather than processing entire files
• Start with Standard mode to preview results before using Pro mode
• Pro mode is ideal for complex medical malpractice or multi-defendant cases
• Summary length: 200 words is optimal for most cases
```

### 7.4 Realistic Time Estimates

**Standard Model (9B):**
* Summary generation: 5-15 minutes (depends on combined document length and summary word count)
* Vocabulary extraction: 2-5 minutes
* Definitions (50 terms): 3-7 minutes
* **Total: 10-27 minutes**

**Pro Model (27B):**
* Summary generation: 30-60 minutes
* Vocabulary extraction: 8-15 minutes
* Definitions (50 terms): 10-20 minutes
* **Total: 48-95 minutes**

**Factors affecting speed:**
* CPU performance (more cores = faster)
* Total token count in combined documents
* Requested summary length
* Number of vocabulary terms requiring definitions

### 7.5 Prompt Engineering

#### 7.5.1 Summary Prompt

**Purpose:** Generate ONE cohesive case-level summary from multiple documents.

**Prompt Template:**
```
You are assisting a court reporter who needs a case overview. Multiple legal documents are provided below (complaint, answer, bill of particulars, motions, exhibits, etc.). Each document is marked with a separator.

Synthesize information across ALL documents to create ONE coherent case summary of approximately {USER_WORD_COUNT} words.

Focus on:
- Main plaintiff(s) and defendant(s)
- Core legal claims (e.g., negligence, breach of contract, medical malpractice)
- Defendant's defenses or counterclaims (if mentioned in answer or motion)
- Key factual events and timeline (e.g., dates of injury, surgery, incident)

Be concise, factual, and use plain language. Ignore metadata like page numbers, filing dates, and docket numbers. If documents contain conflicting information, note it briefly.

Do not list documents individually - synthesize them into one narrative.

{COMBINED_DOCUMENT_TEXT}

Now provide the case summary:
```

**Token Budget:**
* Prompt overhead: ~150 tokens
* Input documents: 4000-6000 tokens (after truncation if needed)
* Output summary: 
  * 100 words = ~133 tokens
  * 200 words = ~267 tokens
  * 500 words = ~667 tokens
* **Total: 4300-6800 tokens** (safely within 8K context window)

**Implementation Notes:**
* Replace `{USER_WORD_COUNT}` with slider value
* Replace `{COMBINED_DOCUMENT_TEXT}` with output from truncation logic (section 6.4)
* Strip any extra whitespace from combined text to save tokens

#### 7.5.2 Vocabulary Extraction Prompt

**Purpose:** Extract only rare/unusual terms and proper nouns from combined case documents.

**Prompt Template:**
```
Extract ONLY unusual or specialized terms from the provided legal documents below. Include:

1. Proper nouns: People's names, hospitals, law firms, specific locations, company names
2. Technical/medical terms: Medical procedures, diagnoses, specialized legal terminology, drug names, medical equipment

EXCLUDE common words and basic legal vocabulary (court, plaintiff, defendant, attorney, judge, lawsuit, claim, motion, exhibit, etc.)

Rules:
- NO duplicates (case-insensitive)
- NO possessives (convert "Smith's" to "Smith")
- Use title case
- Include multi-word proper nouns as single entries (e.g., "New York Supreme Court")

Output format: Python list of strings ONLY. No explanations, no other text.

Example output format:
['Dr. Sarah Martinez', 'lumbar discectomy', 'Lenox Hill Hospital', 'subarachnoid hemorrhage', 'acetaminophen', 'Northwell Health']

{COMBINED_DOCUMENT_TEXT}

Now provide the vocabulary list:
```

**Token Budget:**
* Prompt overhead: ~140 tokens
* Input documents: 4000-6000 tokens (same text as summary)
* Output vocabulary list (50 terms avg): ~100 tokens
* **Total: 4240-6240 tokens**

**Post-Processing (Client-Side):**
```python
import re
import json

def parse_vocab_list(ai_output):
    """
    Extract Python list from AI output, handling various formats.
    """
    # Try to find list in output
    match = re.search(r'\[.*\]', ai_output, re.DOTALL)
    if match:
        try:
            vocab_list = json.loads(match.group(0))
            return vocab_list
        except json.JSONDecodeError:
            # Fallback: parse as Python literal
            try:
                vocab_list = eval(match.group(0))
                return vocab_list
            except:
                return None
    return None

def filter_vocab_with_frequency(vocab_list, google_freq_list, threshold=258000):
    """
    Filter vocab to only include rare words (not in top 258k most common).
    Also keep all proper nouns (capitalized).
    """
    filtered = []
    for term in vocab_list:
        # Always keep proper nouns (starts with capital)
        if term[0].isupper():
            filtered.append(term)
        # Keep rare words (below frequency threshold)
        elif term.lower() not in google_freq_list[:threshold]:
            filtered.append(term)
    
    return filtered

def apply_user_exclusions(vocab_list, excluded_terms_file):
    """
    Remove terms that user has manually excluded.
    """
    with open(excluded_terms_file, 'r') as f:
        excluded = set(line.strip().lower() for line in f)
    
    return [term for term in vocab_list if term.lower() not in excluded]
```

**Error Handling:**
```python
# If parsing fails on first attempt
if vocab_list is None:
    # Retry with simpler prompt
    simplified_prompt = """
    List unusual words from this text, one per line. Include only proper nouns and technical terms.
    
    {COMBINED_DOCUMENT_TEXT}
    """
    # Try again with simplified prompt
    
# If second attempt fails
if vocab_list is None:
    # Skip vocabulary generation, still show summary
    show_error("⚠ Vocabulary extraction failed. Summary generated successfully.")
    vocab_list = []
```

#### 7.5.3 Definition Prompt (Per Term)

**Purpose:** Generate concise definitions for terms in the vocabulary list.

**Prompt Template:**
```
Define the term '{TERM}' in one concise sentence (10-15 words maximum).

If it's a proper noun (person, place, organization), respond with: 'Proper noun - [type]'
Examples:
- 'Proper noun - hospital name'
- 'Proper noun - person's name'
- 'Proper noun - medical practice'

Be direct and factual only. No extra explanation.
```

**Token Budget (Per Term):**
* Prompt + term: ~30 tokens
* Output definition: ~20 tokens
* **Total per term: ~50 tokens**

**Batch Processing:**
* Process definitions sequentially (not all at once)
* Update progress: "Generating definitions... (12/50)"
* Cancel button available during this phase

**Caching Strategy:**
```python
# Cache definitions locally to avoid re-processing common terms
# %APPDATA%/LocalScribe/cache/definitions.json
{
  "mri": "Medical imaging technique using magnetic fields to visualize internal body structures.",
  "deposition": "Proper noun - legal term",  # User might manually edit this
  "dr. john smith": "Proper noun - person's name"
}

def get_definition(term, cache, model, use_ai=True):
    """
    Get definition from cache or generate with AI.
    """
    term_lower = term.lower()
    
    # Check cache first
    if term_lower in cache:
        return cache[term_lower]
    
    # Generate with AI if enabled
    if use_ai:
        definition = generate_definition_with_ai(term, model)
        # Cache for future use
        cache[term_lower] = definition
        save_cache(cache)
        return definition
    else:
        return "No definition available"
```

### 7.6 Streaming Implementation

**Purpose:** Display generated text in real-time so users see progress and can cancel if output is incorrect.

**Benefits of Streaming:**
* Users see immediate feedback (text appearing word-by-word)
* Can cancel generation early if output is wrong
* Better UX during long waits (reduces perceived time)
* Easier debugging (see exactly where generation failed)

---

## 8. Output Generation & Display

### 8.1 Summary Display

**UI Component:** Read-only `QTextEdit` with rich text formatting

**Features:**
* Word-wrapped, scrollable text area
* "Copy to Clipboard" button
* "Save as TXT" button → saves to `Documents/LocalScribe/Summaries/case_summary_YYYY-MM-DD_HHMMSS.txt`
* Character count displayed: "Summary: 247 words"

### 8.2 Vocabulary List Display & Management

**UI Component:** `QTableWidget` with 4 columns

| Term | Category | Definition | Actions |
|------|----------|------------|---------|
| Dr. Robert Chen | Proper Noun | Proper noun - person's name | 🗑️ |
| lumbar discectomy | Technical | Surgical removal of herniated disc material in lower spine. | 🗑️ |
| Lenox Hill Hospital | Proper Noun | Proper noun - hospital name | 🗑️ |
| spondylosis | Technical | Degenerative condition affecting spinal discs and joints. | 🗑️ |

**Columns:**
1. **Term** (sortable alphabetically)
2. **Category:** "Proper Noun" or "Technical"
3. **Definition:** 
   * From AI if generated
   * "No definition available" if Standard mode (no AI definitions)
   * "(AI-Generated)" label appended for AI definitions in Pro mode
4. **Actions:** 
   * 🗑️ "Hide" button → adds term to exclusion list

**Features:**
* Sortable by column (click headers)
* Filter dropdown: "Show All" / "Proper Nouns Only" / "Technical Terms Only"
* "Export as CSV" button → saves to `Documents/LocalScribe/Vocabulary/vocab_YYYY-MM-DD_HHMMSS.csv`
* Row count displayed: "50 terms found"

**User Exclusion System:**
```python
# When user clicks "Hide" button
def exclude_term(term):
    # Add to local exclusion file
    exclusion_file = os.path.join(APPDATA, "LocalScribe", "excluded_vocab.txt")
    with open(exclusion_file, 'a') as f:
        f.write(term.lower() + '\n')
    
    # Remove from current display
    table.removeRow(term_row_index)
    
    # Show confirmation
    status_bar.showMessage(f"'{term}' will be excluded from future vocabulary lists", 3000)

# Load exclusions on startup
def load_exclusions():
    exclusion_file = os.path.join(APPDATA, "LocalScribe", "excluded_vocab.txt")
    if os.path.exists(exclusion_file):
        with open(exclusion_file, 'r') as f:
            return set(line.strip().lower() for line in f)
    return set()
```

**Vocabulary Settings (in Preferences):**
* "Reset Exclusion List" button → clears all user exclusions with confirmation
* "Edit Exclusion List" button → opens text file in system editor
* Display count: "Currently excluding 23 terms"

### 8.3 CSV Export Format

**File structure:**
```csv
Term,Category,Definition,Source
Dr. Robert Chen,Proper Noun,Proper noun - person's name,AI
lumbar discectomy,Technical,Surgical removal of herniated disc material in lower spine.,AI
Lenox Hill Hospital,Proper Noun,Proper noun - hospital name,AI
spondylosis,Technical,Degenerative condition affecting spinal discs and joints.,AI
NYSCEF,Technical,No definition available,None
```

**Export Implementation:**
```python
import csv

def export_vocabulary_csv(vocab_table, output_path):
    """
    Export vocabulary table to CSV.
    """
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Header
        writer.writerow(['Term', 'Category', 'Definition', 'Source'])
        
        # Data rows
        for row_idx in range(vocab_table.rowCount()):
            term = vocab_table.item(row_idx, 0).text()
            category = vocab_table.item(row_idx, 1).text()
            definition = vocab_table.item(row_idx, 2).text()
            
            # Determine source
            if "No definition" in definition:
                source = "None"
            else:
                source = "AI"
            
            writer.writerow([term, category, definition, source])
    
    return output_path
```

---

## 9. First Priority Tasks for Implementation

### 9.1 Phase 1: Pre-processing Engine (IMPLEMENT FIRST)

**Goal:** Build a standalone, testable `extractor.py` module that can process documents from the command line.

**Requirements:**
```bash
# Command line interface
python extractor.py --input file1.pdf file2.pdf --output-dir ./cleaned --jurisdiction ny

# Output:
# - Creates ./cleaned/file1_cleaned.txt
# - Creates ./cleaned/file2_cleaned.txt
# - Prints summary report to console
```

**Module Structure:**
```python
# src/extraction/raw_text_extractor.py
class RawTextExtractor:
    def __init__(self, jurisdiction="ny"):
        self.jurisdiction = jurisdiction
        self.legal_keywords = self.load_keywords()
        self.english_words = self.load_dictionary()

    def process_document(self, file_path):
        """
        Process a single document (extraction + basic normalization).
        Returns: (extracted_text, confidence, method, status, error_message)
        """
        pass

    def sniff_file_type(self, file_path):
        """Detect if PDF, TXT, RTF."""
        pass

    def extract_text_from_pdf(self, pdf_path):
        """Use pdfplumber to extract text."""
        pass

    def calculate_dictionary_confidence(self, text):
        """Calculate % of words that are valid English."""
        pass

    def perform_ocr(self, pdf_path):
        """Run Tesseract OCR and return text + confidence."""
        pass

    def normalize_text(self, raw_text):
        """Apply normalization rules (de-hyphenation, page removal, filtering)."""
        pass

    def load_keywords(self):
        """Load legal keywords for jurisdiction."""
        pass

    def load_dictionary(self):
        """Load NLTK English words."""
        pass

# Test it
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract and normalize text from legal documents")
    parser.add_argument("--input", nargs="+", required=True, help="Input files")
    parser.add_argument("--output-dir", default="./extracted", help="Output directory")
    parser.add_argument("--jurisdiction", default="ny", help="Jurisdiction (ny, ca, federal)")

    args = parser.parse_args()

    extractor = RawTextExtractor(jurisdiction=args.jurisdiction)
    
    results = []
    for file_path in args.input:
        result = extractor.process_document(file_path)
        results.append(result)
        
        # Save cleaned text
        if result['status'] == 'success':
            output_file = os.path.join(args.output_dir, f"{result['filename']}_cleaned.txt")
            with open(output_file, 'w') as f:
                f.write(result['cleaned_text'])
    
    # Print summary report
    print("\n=== PROCESSING SUMMARY ===")
    for result in results:
        print(f"{result['filename']}: {result['status']} - {result['method']} - {result['confidence']}%")
```

**Testing Checklist:**
- [ ] Extract text from digital PDFs correctly
- [ ] Detect scanned PDFs and trigger OCR
- [ ] OCR produces readable text with reasonable confidence
- [ ] Cleaning rules remove junk while preserving content
- [ ] De-hyphenation works correctly
- [ ] Legal headers (ALL CAPS) are preserved
- [ ] Handle password-protected PDFs gracefully
- [ ] Handle corrupted files gracefully
- [ ] Handle files >500MB correctly
- [ ] Process multiple files in batch

**Deliverable:** A working `extractor.py` that can be run independently before UI development begins.

---

## 9.5. Program Flow (Complete Application Workflow)

### Overview

LocalScribe executes a sophisticated multi-phase pipeline that transforms raw legal documents into structured summaries and vocabulary lists. The entire workflow is built on a queue-based threading architecture that keeps the GUI responsive while performing CPU-intensive operations in background threads and separate processes.

**Key Architectural Principles:**
- **Multi-threaded:** Main GUI thread delegates work to background ProcessingWorker thread
- **Queue-based communication:** Inter-thread messages via `ui_queue` prevent blocking
- **Multiprocess AI worker:** Ollama generation runs in isolated process to prevent CPU contention
- **Progressive streaming:** Results appear incrementally as processing completes

---

### Phase 1: GUI Initialization & User Interface Setup

**Files:** `src/main.py`, `src/ui/main_window.py`, `src/ui/quadrant_builder.py`

1. **Application Entry Point** (`src/main.py::main()`, lines 14-68)
   - Add project root to Python path for imports
   - Import `src.ai` module BEFORE CustomTkinter (avoids DirectML DLL conflicts on Windows)
   - Call `setup_file_logging()` to redirect stdout/stderr to timestamped log files in `LOGS_DIR`
   - Enable multiprocessing support for Windows frozen executables
   - Configure CustomTkinter appearance: "System" mode, "blue" theme

2. **Main Window Instantiation** (`src/ui/main_window.py::__init__()`, lines 36-67)
   - Initialize application state variables:
     - `selected_files`: List of file paths chosen by user
     - `processing_results`: Extracted text and metadata from documents
     - `worker`: ProcessingWorker thread for document extraction
     - `ui_queue`: Queue for thread-safe inter-thread communication
     - `ai_worker_manager`: OllamaAIWorkerManager for AI generation
   - Create window layout via `_create_main_layout()`
   - Create menu bar via `_create_menus()`
   - Create toolbar with "Select Files" button via `_create_toolbar()`
   - Create central widget (four-quadrant layout) via `_create_central_widget()`
   - Create status bar with progress indicator via `_create_status_bar()`

3. **Four-Quadrant UI Layout Creation** (`src/ui/quadrant_builder.py`, lines 1-221)
   - **Top-Left Quadrant:** FileReviewTable widget
     - Displays selected documents in treeview table
     - Columns: Include, Filename, Status, Method, Confidence, Pages, Size
     - Status indicators: ✓ (green/ready), ⚠ (yellow/low quality), ✗ (red/error), ○ (gray/pending)
   - **Top-Right Quadrant:** ModelSelectionWidget
     - Dropdown selector for Ollama AI models
     - "Models run locally and offline" info label
     - Auto-detects available models via `ModelManager.list_models()`
   - **Bottom-Left Quadrant:** DynamicOutputWidget
     - Dropdown to select output type (individual summaries, meta-summary, vocab CSV)
     - Text view for summaries or treeview for CSV data
     - "Copy to Clipboard" and "Save to File..." buttons
   - **Bottom-Right Quadrant:** OutputOptionsWidget
     - Summary length slider (100-500 words, default 200)
     - Checkboxes: Individual Summaries, Meta-Summary, Rare Word List CSV
     - "Generate All Outputs" button (disabled until files selected)

4. **Queue Polling Loop Initialization** (`src/ui/main_window.py::__init__()`, line 62)
   - Register `_process_queue()` to run every 100ms via `after(100, _process_queue)`
   - This polling loop checks for messages from background workers

5. **Service Health Check** (`src/ui/main_window.py::__init__()`, line 65)
   - Call `_check_ollama_service()` to verify Ollama is running at http://localhost:11434
   - Display warning if Ollama not found, with platform-specific instructions

---

### Phase 2: File Selection & Document Loading

**Files:** `src/ui/main_window.py`, `src/ui/widgets.py`

1. **User Initiates File Selection**
   - User clicks "Select Files..." button in toolbar
   - Triggers `select_files()` method (lines 141-170)

2. **File Dialog Opens**
   - `filedialog.askopenfilenames()` with filters:
     - `*.pdf` - PDF documents (digital or scanned)
     - `*.txt` - Plain text files
     - `*.rtf` - Rich Text Format documents
   - User selects one or more files, dialog returns list of paths

3. **UI Update & Table Population** (lines 160-170)
   - Update `selected_files` instance variable with chosen paths
   - Update file counter label (e.g., "3 file(s) selected")
   - Enable "Generate All Outputs" button
   - Populate FileReviewTable with pending entries:
     - Call `file_table.add_result(filename, {'status': 'pending', ...})`
     - Display filename, file size (in bytes), page count (if available)
     - Status shows "○" (pending) with gray background
     - Confidence shows 0% initially

---

### Phase 3: Document Processing Pipeline

**Files:** `src/extraction/raw_text_extractor.py`, `src/sanitization/character_sanitizer.py`, `src/ui/workers.py`

#### **Phase 3A: Raw Text Extraction (Steps 1-2)**

1. **Processing Initiation** (`src/ui/main_window.py::_start_generation()`, lines 172-198)
   - User clicks "Generate All Outputs" button
   - Validate: Files selected, model chosen, at least one output type checked
   - Collect configuration:
     - `model_name`: Selected Ollama model
     - `summary_length`: Word count from slider (100-500)
     - `output_options`: Dict with checkboxes for individual_summaries, meta_summary, vocab_csv
   - Store in `pending_ai_generation` for later use
   - Call `start_processing()` (lines 200-221)

2. **Background Worker Thread Creation** (`src/ui/main_window.py::start_processing()`, lines 200-221)
   - Disable "Select Files" and "Generate" buttons (prevent duplicate processing)
   - Show progress bar (0%)
   - Create ProcessingWorker thread: `ProcessingWorker(selected_files, self.ui_queue)`
   - Start worker thread (background execution, non-blocking)

3. **File Type Detection** (`src/extraction/raw_text_extractor.py::process_document()`, lines 183-195)
   - Get file extension: `.pdf`, `.txt`, or `.rtf`
   - Route to appropriate processing method
   - TXT/RTF → direct read
   - PDF → proceed to text extraction

4. **Text Extraction from PDF** (`src/extraction/raw_text_extractor.py::_process_pdf()`, lines 283-319)
   - Attempt digital text extraction using `pdfplumber.open(pdf_path)`
   - Extract text from each page: `page.extract_text() or ""`
   - Concatenate all pages into single string
   - Calculate dictionary confidence score (next step)

5. **Confidence Scoring & OCR Decision** (`src/extraction/raw_text_extractor.py::_calculate_dictionary_confidence()`, lines 476-501)
   - Tokenize extracted text by whitespace
   - Load English words set from NLTK corpus: `nltk.corpus.words.words()`
   - Count tokens matching dictionary (case-insensitive)
   - Calculate percentage: `(valid_words / total_words) * 100`
   - **Decision threshold:** If confidence < 60%, PDF is considered scanned → trigger OCR
   - Otherwise, use digital text extraction

6. **OCR Processing (if confidence < 60%)** (`src/extraction/raw_text_extractor.py::_perform_ocr()`, lines 392-437)
   - Convert PDF to images: `pdf2image.convert_from_path(pdf_path, dpi=300)`
   - For each image, apply Tesseract: `pytesseract.image_to_string(image)`
   - Concatenate all OCR results
   - Recalculate confidence using same dictionary method
   - Update processing method: "ocr" instead of "digital_text"

7. **Basic Text Normalization** (`src/extraction/raw_text_extractor.py::_normalize_text()`, lines 503-576)
   - **De-hyphenation:** Rejoin split words: `defen-\ndant` → `defendant`
     - Regex: `re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)`
     - Preserves legitimate hyphens (attorney-client)
   - **Page Number Removal:** Delete lines that are just page markers
     - Patterns: `Page 1`, `- 2 -`, `P. 3`, `[10]`
     - Method: `_is_page_number()` returns True/False
   - **Line Filtering:** Remove low-quality lines
     - Keep lines with length > 15 characters
     - Require either lowercase letters OR legal keywords (from `LEGAL_KEYWORDS_NY`)
     - Filter out lines with more symbols than alphabetic characters
   - **Whitespace Normalization:** Remove excess blank lines
     - Regex: `re.sub(r'\n\s*\n\s*\n+', '\n\n', text)`
     - Max 1 blank line between paragraphs

8. **Result Dictionary Creation**
   - Return dict with extracted document metadata:
     ```python
     {
         'filename': 'complaint.pdf',
         'file_path': '/path/to/complaint.pdf',
         'status': 'success',  # or 'warning' if confidence < threshold
         'method': 'ocr',  # or 'digital_text'/'direct_read'
         'confidence': 87,  # 0-100 OCR confidence percentage
         'extracted_text': '...full document text...',
         'page_count': 45,
         'file_size': 2457600,  # bytes
         'case_numbers': ['2025-CV-12345'],  # detected case IDs
         'error_message': None
     }
     ```
   - Send to UI queue: `ui_queue.put(('file_processed', result))`
   - Update FileReviewTable: filename changes from "○ pending" to "✓ success" or "⚠ warning"

#### **Phase 3B: Character Sanitization (Step 2.5)**

**File:** `src/sanitization/character_sanitizer.py`

This 6-stage pipeline fixes Unicode and encoding corruption discovered in extracted text (e.g., from debug_flow.txt: `ñêcessary`, `dccedêñt`, `Defeñdañt`).

1. **Mojibake Recovery** (`CharacterSanitizer._fix_mojibake()`, lines 113-132)
   - Use `ftfy.fix_text(text)` to recover text with encoding corruption
   - Examples: `ñêcessary` → `necessary`, `Defeñdañt` → `Defendant`
   - ftfy applies heuristics to detect and fix common encoding issues

2. **Unicode Normalization** (`CharacterSanitizer._normalize_unicode()`, lines 134-149)
   - Apply NFKC normalization: `unicodedata.normalize('NFKC', text)`
   - Decomposes characters to base forms, then recomposes
   - Handles ligatures, superscripts, and compatibility characters

3. **Accent Transliteration** (`CharacterSanitizer._transliterate_text()`, lines 151-175)
   - Use `unidecode.unidecode(text)` to convert accented characters to ASCII
   - Examples: `é` → `e`, `ê` → `e`, `ñ` → `n`
   - Fixes stray accents from OCR corruption

4. **Redaction Handling** (`CharacterSanitizer._handle_redactions()`, lines 177-198)
   - Detect redaction block patterns: `█`, `▓`, `▒`
   - Replace sequences with marker: `██` → ` [REDACTED] `
   - Preserves document meaning while removing sensitive content

5. **Control Character & Private-Use Removal** (`CharacterSanitizer._clean_problematic_chars()`, lines 200-260)
   - Iterate through each character
   - Use `unicodedata.category(char)` to classify:
     - Control chars (Cc): Non-breaking spaces, format chars
     - Private-use (Co): Invalid codepoints
     - Surrogates (Cs): Malformed UTF-8 sequences
   - Remove or replace with spaces; preserve newlines/tabs

6. **Whitespace Cleanup** (`CharacterSanitizer._clean_whitespace()`, lines 262-284)
   - Replace tabs with spaces
   - Replace non-breaking spaces with regular spaces
   - Collapse multiple spaces: `r' {2,}'` → single space
   - Collapse excess blank lines: `r'\n{3,}'` → `\n\n`

**Returns:** `(cleaned_text, sanitization_stats)` where stats dict contains:
```python
{
    'mojibake_fixed': 15,
    'transliterations': 42,
    'redactions_replaced': 3,
    'control_chars_removed': 8,
    'chars_removed': 68
}
```

#### **Phase 3C: Smart Preprocessing (Step 3) [PLANNED - NOT YET IMPLEMENTED]**

**Planned File:** `src/preprocessing/pipeline.py` (design in scratchpad.md, lines 298-349)

This phase will remove OCR/document artifacts and support user customization:

1. **Title Page Removal** (TitlePageRemover)
   - Detect and remove metadata-heavy first sections
   - Identifies: case numbers, filing dates, firm names, page counts
   - Preserves actual content

2. **Line Number Removal** (LineNumberRemover)
   - Detect margins with sequential line numbers: 1, 2, 3... 123, 124...
   - Remove without destroying text content

3. **Header/Footer Removal** (HeaderFooterRemover)
   - Identify repetitive page headers/footers
   - Patterns: case name, page X of Y, date stamps
   - Remove while preserving unique content

4. **Q&A Format Conversion** (QAConverter)
   - Detect deposition Q&A format: `Q. question text` / `A. answer text`
   - Expand to plain English: `Question: question text` / `Answer: answer text`
   - Improves LLM readability

5. **User Customization Hooks** (Planned)
   - Support custom Python functions: `(str) → str`
   - Allow users to add domain-specific preprocessing steps
   - Configuration: YAML or Python function registry

---

### Phase 4: Output Generation (Parallel Workflows)

#### **Workflow A: Vocabulary Extraction [FULLY OPERATIONAL - Session 44]**

**Files:** `src/vocabulary/vocabulary_extractor.py`, `src/extraction/llm_extractor.py`, `src/vocabulary/reconciler.py`, `src/ui/widgets.py`

**Current Status:** Dual NER + LLM extraction with reconciliation is fully operational. Shows "Found By" column indicating source (Both, NER, LLM).

1. **User Configuration**
   - User checks "Rare Word List (CSV)" checkbox in OutputOptionsWidget
   - Indicates vocabulary extraction is desired

2. **Vocabulary Extraction Call** (not yet integrated)
   - Would be called after document processing with combined text
   - `extractor = VocabularyExtractor()`
   - `vocab_list = extractor.extract(combined_text)`

3. **NLP Parsing** (`src/vocabulary_extractor.py::extract()`, lines 95-127)
   - Load spaCy English model: `spacy.load('en_core_web_sm')`
   - Parse text through NLP pipeline
   - Extract multi-word named entities: PERSON, ORG, GPE (location)
   - Count token frequencies (case-insensitive)

4. **Unusual Word Filtering** (`src/vocabulary_extractor.py::_is_unusual()`)
   - Exclude common words (found in WordNet)
   - Exclude legal terms (from config/legal_exclude.txt)
   - Include named entities (always)
   - Include medical terms (from config/medical_terms.txt)
   - Include acronyms: 2+ uppercase letters (CT, FDA, HIPAA)
   - Include other rare tokens

5. **Categorization & Relevance Assignment** (lines 132-176)
   - **Categories:**
     - "Acronym" - All uppercase, 2+ chars
     - "Medical Term" - In medical_terms.txt
     - "Proper Noun (Person/Org/Location)" - spaCy entity type
     - "Technical Term" - Default for rare words
   - **Relevance scoring:**
     - Proper Nouns: High (Very High if frequency > 1)
     - Medical Terms: Medium (High if frequency ≥ 2)
     - Acronyms: Medium (High if frequency ≥ 2)
     - Technical Terms: Low (Medium if frequency ≥ 3)

6. **Definition Lookup** (via NLTK WordNet)
   - For each term, attempt to find definition in WordNet
   - Return first definition found, or "N/A" if not found
   - Note: No AI-generated definitions in current implementation

7. **CSV Data Preparation**
   - Format as list of lists (header + rows):
     ```python
     [
         ['Term', 'Category', 'Relevance to Case', 'Definition'],
         ['cardiomyopathy', 'Medical Term', 'Medium', 'a disease of the heart muscle'],
         ['Dr. Chen', 'Proper Noun (Person)', 'High', 'N/A'],
         ...
     ]
     ```
   - Would be sent to UI via queue: `ui_queue.put(('vocab_csv_generated', csv_data))`
   - **Note:** Integration step missing - not currently called in workers.py

#### **Workflow B: Progressive Summarization [FULLY OPERATIONAL]**

**Files:** `src/progressive_summarizer.py`, `src/chunking_engine.py`, `src/ai/ollama_model_manager.py`, `src/ui/ollama_worker.py`

This sophisticated workflow handles documents of arbitrary length using intelligent chunking and progressive (rolling) summaries.

1. **Document Chunking** (`ChunkingEngine.chunk_document()`, via ProgressiveSummarizer)
   - Split long documents into manageable chunks
   - Chunk constraints:
     - Soft max: 2000 words per chunk
     - Hard max: 3000 words
     - Min: 500 words
   - Prefer splitting at section boundaries (detected via regex patterns)
   - Return list of Chunk objects with: `{chunk_num, text, section_name, word_count}`

2. **Batch Boundary Calculation** (ProgressiveSummarizer)
   - Determine when to update the progressive (rolling) summary
   - Three strategies in priority order:
     - **Section-aware:** Update at section boundaries (min 3, max 15 chunks per batch)
     - **Adaptive:** Early doc (chunks 1-20): every 5 chunks; middle (21-80): every 10; late: every 5
     - **Fixed:** Update every 5 chunks (fallback)

3. **DataFrame Preparation** (ProgressiveSummarizer)
   - Create pandas DataFrame to track processing state:
     ```python
     columns: [chunk_num, chunk_text, chunk_summary, progressive_summary,
               section_detected, word_count, processing_time_sec]
     ```

4. **Context Building for Each Chunk** (`ProgressiveSummarizer.get_context_for_chunk()`)
   - **Global context:** Previous progressive summary
     - 1-2 sentences providing document overview
     - Example: "[Document overview: Plaintiff alleges negligence. Defendant denies.]"
   - **Local context:** Summary of immediately previous chunk
     - 1-2 sentences providing narrative continuity
     - Example: "[Previous: Witness testified about accident scene.]"

5. **AI Prompt Generation** (ProgressiveSummarizer)
   - Load template from `config/chunked_prompt_template.txt`
   - Populate with:
     - Document overview (global context)
     - Previous section summary (local context)
     - Current section text
     - Target word range (e.g., 52-97 words)
   - Parameters: min_words, max_words, focus points

6. **Ollama AI Generation** (`src/ai/ollama_model_manager.py::generate()`)
   - **Multiprocess worker:** Separate process runs `ollama_generation_worker_process()` (src/ui/ollama_worker.py)
     - Prevents GUI blocking during long generation
   - **API call:** POST to `http://localhost:11434/api/generate`
     - Model: Selected by user (gemma3:1b, llama2:7b, etc.)
     - Prompt: Wrapped in model-specific format (Phase 2.7)
     - Temperature: 0.7 (configurable)
     - Top-p: 0.9 (configurable)
     - Max tokens: Calculated from word count budget
   - **Response:** Generated text + token count + timing info

7. **Progressive Summary Update** (at batch boundaries)
   - Collect all chunk summaries since last update
   - Generate meta-summary combining them
   - Store in DataFrame `progressive_summary` column
   - This becomes global context for next batch of chunks

8. **Individual Document Summary** (per document)
   - Concatenate all chunk summaries for single document
   - Could optionally generate final meta-summary for that document alone

9. **Meta-Summary Across All Documents** (post-processing)
   - If multiple documents processed, generate aggregate summary
   - Combines key points from all individual document summaries
   - Provides high-level case overview

10. **Result Transmission to GUI** (via queue)
    - Send completion message: `ui_queue.put(('summary_result', {'summary': text, 'meta_summary': text}))`
    - QueueMessageHandler routes to `handle_summary_result()`
    - Update DynamicOutputWidget with results

---

### Phase 5: Results Display & User Export

**Files:** `src/ui/dynamic_output.py`, `src/ui/main_window.py`, `src/ui/queue_message_handler.py`

1. **Queue Message Routing** (`src/ui/queue_message_handler.py::process_message()`, lines 34-71)
   - Receive messages from worker threads: `('message_type', data)`
   - Route to appropriate handler:
     - `('progress', ...)` → `handle_progress()`
     - `('file_processed', ...)` → `handle_file_processed()`
     - `('processing_finished', ...)` → `handle_processing_finished()`
     - `('summary_result', ...)` → `handle_summary_result()`
     - `('vocab_csv_generated', ...)` → `handle_vocab_csv_generated()`
     - `('error', ...)` → `handle_error()`

2. **Output Widget Display** (`src/ui/dynamic_output.py::DynamicOutputWidget`)
   - Dropdown selector to choose output type:
     - Individual summaries (one per file)
     - Meta-summary (aggregate across files)
     - Rare Word List CSV (if vocabulary extraction enabled)
   - Display area:
     - **For text:** CustomTkinter CTkTextbox (read-only, word-wrapped)
     - **For CSV:** Treeview widget with sortable columns

3. **User Copy/Save Operations**
   - **Copy to Clipboard:** `CTkMessagebox` → copy text to system clipboard
   - **Save to File:** `filedialog.asksaveasfilename()`
     - Summaries: Save as `.txt` to `Documents/LocalScribe/Summaries/`
     - CSV: Save as `.csv` to `Documents/LocalScribe/Vocabulary/`
     - Filename pattern: `summary_YYYY-MM-DD_HHMMSS.txt`

4. **Status Bar Updates**
   - Progress bar → 100%
   - Status message: "Summary generation complete!"
   - Re-enable "Select Files" and "Generate" buttons

---

### Architecture & Communication Patterns

#### **Queue-Based Threading Architecture**

```
Main GUI Thread                    Background Worker Thread
─────────────────────────────────────────────────────────────
User clicks button
    ↓
start_processing()
    ├─ disable buttons
    ├─ create ProcessingWorker()
    └─ start worker thread ────→  ProcessingWorker.run()
                                      ├─ for file in files:
_process_queue() [loops every 100ms]  │   RawTextExtractor.process_document()
    ├─ check ui_queue              │   sanitizer.sanitize()
    ├─ receive ('file_processed')  │   ui_queue.put(('file_processed', result))
    │   file_table.update_row()    └─ ui_queue.put(('processing_finished', ...))
    │
    └─ receive ('processing_finished')
        ├─ combine extracted text
        └─ _start_ai_generation()
           ai_worker_manager.send_task()
```

#### **Multiprocess AI Worker Pattern**

```
Main GUI Process                   Separate AI Worker Process
────────────────────────────────────────────────────────────
_start_ai_generation()
    ├─ create OllamaAIWorkerManager
    └─ send_task() ───────────→    ollama_generation_worker_process()
                                       ├─ connect to Ollama (http://localhost:11434)
_process_queue() [loops]          └─ POST /api/generate
    ├─ check ai_worker_manager        ├─ streaming=false (avoids UTF-8 issues)
    └─ receive result ←───────────    └─ return generated text
        handle_summary_result()
```

#### **Data Flow Example**

```
User File: complaint.pdf (45 pages, 2.3 MB)
    ↓
RawTextExtractor.process_document()
    ├─ Digital text extraction: 98% confidence
    └─ Result: 15,000 tokens
        ↓
    CharacterSanitizer.sanitize()
    ├─ Fixed mojibake: 12 chars
    ├─ Removed control chars: 5
    └─ Clean text: 14,998 tokens
        ↓
    [Phase 3C would go here - not yet implemented]
        ↓
    ProgressiveSummarizer
    ├─ Chunks into 8 sections (2000 words each)
    ├─ Generates 8 chunk summaries (75 words each)
    ├─ Updates progressive summary at 3 batch boundaries
    └─ Final summary: 200 words (user-requested)
        ↓
    UI Queue Message: ('summary_result', {'summary': '...'})
        ↓
    DynamicOutputWidget displays summary
    User clicks "Save to File..." → saves as complaint_2025-11-25_143022.txt
```

---

### Implementation Status Summary

| Phase | Component | Status | Notes |
|-------|-----------|--------|-------|
| 1 | GUI Initialization | ✅ Complete | CustomTkinter, 4-quadrant layout, responsive |
| 2 | File Selection | ✅ Complete | File dialog, table population |
| 3A | Raw Text Extraction | ✅ Complete | PDF/TXT/RTF, OCR, confidence scoring (Session 1) |
| 3B | Character Sanitization | ✅ Complete | 6-stage cleanup, ftfy+unidecode (Session 3) |
| 3C | Smart Preprocessing | 📋 Planned | Design ready (scratchpad), waiting implementation |
| 4A | Vocabulary Extraction | ✅ Complete | Dual NER + LLM extraction with reconciliation, "Found By" column (Session 44) |
| 4B | Summarization | ✅ Complete | Progressive chunking, Ollama integration, working end-to-end |
| 4C | Q&A Query Transform | ✅ Complete | LlamaIndex + Ollama query expansion for vague questions (Session 44) |
| 5 | Results Display | ✅ Complete | CSV/text display, copy/save functionality |

---

## 10. Development Phases Summary

### Phase 1: Pre-processing Engine (2-3 weeks)
**Deliverable:** Command-line `extractor.py` module

### Phase 2: Basic UI Shell (2 weeks)
**Deliverable:** CustomTkinter main window with file selection and preprocessing integration

### Phase 3: AI Integration (2-3 weeks)
**Deliverable:** Model loading, streaming generation, summary display

### Phase 4: Vocabulary & Definitions (1-2 weeks)
**Deliverable:** Vocab extraction, filtering, definitions, table display

### Phase 5: License System Integration (1 week)
**Deliverable:** License validation, model downloads, quota tracking

### Phase 6: Settings & Polish (1-2 weeks)
**Deliverable:** Document priority config, preferences, final error handling

### Phase 7: Packaging & Distribution (1 week)
**Deliverable:** PyInstaller executable, installer, documentation

**Total Estimated Timeline: 10-14 weeks**

---

## 11. System Requirements

### Minimum Requirements
* **CPU:** Intel Core i5 / AMD Ryzen 5 (4+ cores)
* **RAM:** 16 GB (8 GB minimum but slow)
* **Storage:** 50 GB free space
* **OS:** Windows 10/11 (64-bit)

### Recommended Requirements
* **CPU:** Intel Core i7/i9 / AMD Ryzen 7/9 (8+ cores)
* **RAM:** 32 GB
* **Storage:** 100 GB free (SSD)
* **OS:** Windows 11 (64-bit)

---

## 12. Code Patterns & Conventions

### 12.1 Transformation Pipeline Logging Pattern

**Pattern:** All transformation stages use variable reassignment (`text = transform(text)`) with comprehensive logging for observability.

**Rationale:**
- **Pythonic:** Trusts Python's automatic garbage collection (standard practice)
- **Simple Code:** No verbose memory management clutters transformation logic
- **Better Observability:** Comprehensive logging shows exactly what happened at each stage
- **Performance Insights:** Timing data for each stage helps identify bottlenecks
- **Debugging Support:** Success/failure logs make troubleshooting easier

**Convention:**
```python
# ❌ BAD: No logging, unclear what happened
text = transform(text)

# ✅ GOOD: Logged transformation with metrics
debug("Starting Stage N: [Stage Name]")
start_time = time.time()
original_len = len(text)

try:
    text = transform(text)
    duration = time.time() - start_time
    debug(f"  ✅ SUCCESS ({duration:.3f}s) - [Action summary]")
    debug(f"     Input: {original_len} | Output: {len(text)} | Delta: {len(text) - original_len:+d}")
except Exception as e:
    duration = time.time() - start_time
    debug(f"  ❌ FAILED ({duration:.3f}s) - {type(e).__name__}: {str(e)}")
    raise
```

**Logging Categories:**

1. **Execution Tracking**
   - "Starting Stage N: [Stage Name]"
   - "✅ SUCCESS: [Stage Name]"
   - "❌ FAILED: [Stage Name]"

2. **Performance Timing**
   - Duration in seconds with 3 decimal places (e.g., "0.125s", "1.234s")
   - Included in both success and failure messages
   - Helps identify slow stages for optimization

3. **Text Metrics**
   - Input character count
   - Output character count
   - Delta (difference) with +/- sign to show growth/shrinkage
   - Item counts for line-based operations (e.g., "Removed 5 page markers")

4. **Error Details**
   - Exception type (e.g., "ValueError", "IOError")
   - Exception message for context
   - Always re-raised to stop processing and propagate to caller

**Current Pipeline (11 Stages):**

| Stage | Transformation | Module | Method |
|-------|---|--------|--------|
| 1 | Extract text from file | RawTextExtractor | `process_document()` |
| 2 | De-hyphenation | RawTextExtractor | `_normalize_text()` |
| 3 | Page number removal | RawTextExtractor | `_normalize_text()` |
| 4 | Line filtering | RawTextExtractor | `_normalize_text()` |
| 5 | Whitespace normalization | RawTextExtractor | `_normalize_text()` |
| 6 | Mojibake recovery (ftfy) | CharacterSanitizer | `sanitize()` |
| 7 | Unicode normalization (NFKC) | CharacterSanitizer | `sanitize()` |
| 8 | Transliteration (optional) | CharacterSanitizer | `sanitize()` |
| 9 | Redaction handling | CharacterSanitizer | `sanitize()` |
| 10 | Problematic character removal | CharacterSanitizer | `sanitize()` |
| 11 | Final whitespace cleanup | CharacterSanitizer | `sanitize()` |

### 12.2 Memory Management Strategy

**Approach:** Trust Python's garbage collection with comprehensive logging.

**Rationale:**
- Python's garbage collection (GC) handles automatic memory cleanup
- Logging provides observability without verbose memory management
- For large files, GC is sufficient; explicit `del` is un-Pythonic
- Variable reassignment is standard Python idiom

**When Memory Matters:**
- For files >100MB, monitor performance via debug logs
- If memory is critical, add monitoring/profiling (don't add explicit `del`)
- GC will free memory after each stage completes
- Logging shows timing; high durations may indicate GC pressure

### 12.3 Helper Methods - Keep Generic Signatures

**Rule:** Only orchestration-level code uses the logging pattern above.

Helper methods maintain generic signatures to stay reusable:

```python
# ✅ CORRECT: Generic helper signature
def _fix_mojibake(self, text: str) -> Tuple[str, int]:
    """Fix encoding corruption using ftfy."""
    original = text
    text = ftfy.fix_text(text)
    return text, fixes

# ❌ WRONG: Overly specific - hard to test and reuse
def _fix_mojibake(self, text_rawExtracted: str) -> Tuple[str, int]:
    # Violates separation of concerns
```

**Rationale:**
- Helper methods are implementation details
- Logging/tracking is the caller's responsibility
- Generic parameters are more testable and reusable
- Keeps method signatures stable as patterns evolve

### 12.4 Applying This Pattern to New Stages

When adding new transformation stages (e.g., Phase 3C Smart Preprocessing):

1. **Use variable reassignment**
   ```python
   text = your_transformation(text)
   ```

2. **Add comprehensive logging**
   ```python
   debug("Starting Stage N: [Stage Name]")
   start = time.time()
   original_len = len(text)

   try:
       text = your_transformation(text)
       duration = time.time() - start
       debug(f"  ✅ SUCCESS ({duration:.3f}s) - [Action summary]")
       debug(f"     Input: {original_len} | Output: {len(text)} | Delta: {len(text) - original_len:+d}")
   except Exception as e:
       duration = time.time() - start
       debug(f"  ❌ FAILED ({duration:.3f}s) - {type(e).__name__}: {str(e)}")
       raise
   ```

3. **Update this table** (Section 12.1) to reflect new stages

4. **Document in development_log.md** that new stage follows this pattern

---

**Document Version:** 2.2 (Session 44 - NER+LLM Vocabulary + LlamaIndex Q&A)
**Date:** December 9, 2025
**Status:** Ready for Implementation
