# Security & Privacy Policy

## Data Handling

CasePrepd is designed for processing sensitive legal documents. All processing happens **entirely on your local machine**.

### What stays on your device

- All uploaded documents (PDFs, transcripts, pleadings)
- All extracted text and generated summaries
- All vocabulary analysis and Q&A results
- All ML models (bundled with the application)
- All user preferences and feedback data

### What is NOT sent anywhere

- No document content is transmitted to any cloud service, API, or remote server
- No usage telemetry or analytics are collected
- No network connections are made during document processing

### Local AI Processing

- **Summarization**: Uses Ollama, which runs locally on your machine
- **Named Entity Recognition**: Uses bundled GLiNER and spaCy models (no network required)
- **Embeddings & Retrieval**: Uses bundled Nomic and sentence-transformer models (no network required)
- **OCR**: Uses Tesseract, which runs locally on your machine

### Where your data is stored

| Data | Location |
|------|----------|
| Application files | `C:\Program Files\CasePrepd\` (or chosen install directory) |
| User preferences | `%APPDATA%\CasePrepd\config\` |
| Processing logs | `%APPDATA%\CasePrepd\logs\` |
| ML training data | `%APPDATA%\CasePrepd\data\` |
| Cache files | `%APPDATA%\CasePrepd\cache\` |

All user data remains in your Windows user profile and is not accessible to other users on the same machine.

## Uninstalling

- The Windows uninstaller removes all application files from the install directory
- User data in `%APPDATA%\CasePrepd\` is preserved across reinstalls. To fully remove it, delete that folder manually after uninstalling
- Ollama and Tesseract are separate installations and are not affected by uninstalling CasePrepd

## Reporting Security Issues

If you discover a security vulnerability, please open an issue on the GitHub repository or contact the maintainers directly. Do not include sensitive document content in bug reports.
