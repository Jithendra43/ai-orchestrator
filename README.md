# AI-Orchestrated System for Test Engineering

This intelligent orchestrator analyzes requirements, classifies them, and routes them to the most appropriate specialized agent:
- **Test Case Generator**: Creates structured test cases for software features
- **User Story Generator**: Creates user stories that capture user needs and requirements
- **Functional Requirement Generator**: Creates formal functional requirements
- **Test Data Generator**: Creates comprehensive test data sets linked to test cases

## Features

- Upload and process various document types (PDF, DOCX, TXT, CSV, XLSX)
- Extract business requirements and system architecture details
- Chat interface for generating various software engineering artifacts
- Intelligent agent selection based on query intent
- ID mapping and traceability between artifacts

## Deployment on Streamlit Cloud

### Prerequisites

- A GitHub repository containing this project
- A Streamlit Cloud account
- A GROQ API key

### Deployment Steps

1. **Create a Streamlit Secret**
   - In your Streamlit Cloud dashboard, navigate to your app settings
   - Click on "Secrets" and add your GROQ API key:
     ```toml
     GROQ_API_KEY = "your-groq-api-key-here"
     ```

2. **Link Your GitHub Repository**
   - In Streamlit Cloud dashboard, click "New app"
   - Connect to your GitHub repository
   - Select the main branch
   - Set the main file path to `multi_agent_orchestrator.py`

3. **Advanced Settings (Optional)**
   - Increase memory if needed (recommended at least 1GB)
   - Set Python version to 3.9 or higher

4. **Deploy**
   - Click "Deploy"
   - Wait for the build to complete

### Local Development

To run this application locally:

1. Clone the repository
2. Install requirements:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.streamlit/secrets.toml` file with your API key:
   ```toml
   GROQ_API_KEY = "your-groq-api-key-here"
   ```
4. Run the Streamlit app:
   ```
   streamlit run multi_agent_orchestrator.py
   ```

## Notes

- Large language models require significant memory; adjust resources accordingly
- Streamlit Cloud has a 1GB upload limit; consider this when uploading documents
- The GROQ API has rate limits; monitor usage to avoid disruptions 