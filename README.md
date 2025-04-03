# Course-notes-RAG
A RAG architecture for my course notes. We reccomend using Anaconda Powershell for the following steps.

For this project, we'll be using the following technologies for our architecture.
| Component | Technologies |
|------------|---------|
| Language | Python |
| Large Language Models | Llama 3.2, Mistral 7B  |
| Offline inferences/llm hosting | Ollama |
| Vector Database | Redis Vector DB, Chroma, Pinecone |
| Embedding models | sentence transformers/all-MiniLM-L6-v2, [nomic-embed-text]((https://ollama.com/library/nomic-embed-text)) and [mxbai-embed-large](https://ollama.com/library/mxbai-embed-large) |

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
```
ollama pull mxbai-embed-large
```
```
ollama pull nomic-embed-text
```

5. Set up the Redis container using the Redis setup lecture slides.

# File Architecture:
| File | Utility |
|------------|---------|
| llm_models/llama.py | hosts the LLM class for LLM initialization |
| chroma_vectordb.py | contains the class and methods associated with chromaDB |
| pinecone_vectordb.py | contains the class and methods associated with Pinecone |
| redis_vectordb.py | contains the class and methods associated with Redis VectorDB |
| chunking.py | chunks a given text by chunk size and overlap | 
| embed.py | contains the various embedding model classes and their functions | 
| experiment.py | script to run the experiment and write the results to experiment_results.csv | 
| search_function | a basic search function for the user to interact with the architecture (defaulting to Pinecone + Sentence Transformer) | 
| upload_to_Redis.py | file that contains the script to upload given files/data to the Redis database | 
| upload_to_chroma.py | file that contains the script to upload given files/data to the ChromaDB  | 
| upload_to_pinecone.py | file that contains the script to upload given files/data to the pinecone database |
| visualization.ipynb | Jupyter notebook file that contains python code to graph our findings |  

# How to run the experiment:
1. Ensure all dependencies are downloaded to your virtual environment
2. Ensure that your redis container is set up and running 
3. Run the experiment.py using
```
py experiment.py <PATH_TO_YOUR_FOLDER_CONTAINING_PDF_NOTES>
```
4. The results should be written to experiments_results for you to analyze and manually scored