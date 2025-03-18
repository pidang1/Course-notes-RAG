import ollama

class Llama:
    def __init__(self, model_name="llama3.2"):
        self.model_name = model_name
    
    # Generate response for the given prompt and return the response output
    def generate_response(self, prompt):
        try:
            response = ollama.chat(model=self.model_name, messages=[{"role": "user", "content": prompt}])
            return response["message"]["content"]
        except Exception as e:
            return f"Error generating response: {e}"

# Example usage
llama = Llama()
print(llama.generate_response("Hello! Please give a quick summary about RAG search architecture.")) 
