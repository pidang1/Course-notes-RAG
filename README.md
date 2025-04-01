# Course-notes-RAG
A RAG architecture for my course notes. We reccomend using Anaconda Powershell for the following steps.

For this project, we'll be using the following technologies for our architecture.
| Component | Technologies |
|------------|---------|
| Language | Python |
| Large Language Models | Llama 3.2, Mistral 7B  |
| Offline inferences/llm hosting | Ollama |
| Vector Database | Redis Vector DB, Chroma, Pinecone |
| Embedding models | sentence transformers/all-MiniLM-L6-v2, nomic-embed-text( via [Ollama](https://ollama.com/library/nomic-embed-text)) and InstructorXL |

1. if this is your first time running the project, make sure to run: 
```bash
python -m venv venv
```

2. Then, run this command to activate the virtual environment:
For MacOS:
```bash
source venv/bin/activate
```
For Windows:
```bash
venv\Scripts\activate
```

3. Install requirements needed using this command in the same foler:
```bash
pip install -r requirements.txt
```

4. Ensure you have Ollama installed and both LLM models pulled (llama 3.2 and mistral). To pull the models, run the following on your command line:
```
ollama run llama3.2
```
```
ollama run mistral
```
