def main():
    print("Welcome to RAG search program!")
    searchSettings = []
    try:
        while True:
            menuOptions = {
                "config": "Configure search settings (embedding model, LLM model, database)",
                "search": "Search for information",
                "quit": "Exit the program"
            }
            
            print("\nPlease choose from the following options:")
            for key, value in menuOptions.items():
                print(f"{key}: {menuOptions[key]}")
            user_input = input("\nWhat would you like to do next? : ").strip().lower()
            
            # If config is selected or searchSettings is empty, configure search settings (RAG Architecture)
            if user_input == "config" or searchSettings == []:
                searchSettings = searchConfig()
            # If search is selected, run search
            elif user_input == "search":
                print("Search")
            
            # Exit program if quit is inputted
            elif user_input == "quit":
                print("Exiting program. Goodbye!")
                break
            else:
                print(f"Invalid option. Please choose from {', '.join(menuOptions.keys())}")
    
    except KeyboardInterrupt:
        print("\nExiting, good luck studying!")

# Function to prompt the user for their preferred search configurations (embedding model, LLM model, database)
def searchConfig():
    # Options to choose for embedding model
    embeddingModelOptions = {
        "A": "Option A",
        "B": "Option B",
        "C": "Option C"
    }
    
    # Options to choose for LLM Model
    llmModelOptions = {
        "A": "Option A",
        "B": "Option B",
        "C": "Option C"
    }
    # Options to choose for DB
    databaseOptions = {
        "redis": "Redis Vector DB",
        "chroma": "Chroma",
        "pinecone": "Pinecone"
    }
    
    print("\nPlease select a model for embedding:")
    for option in embeddingModelOptions:
        print(f"{option}: {embeddingModelOptions[option]}")  
    embeddingModel = input("\nYour choice: ").strip().upper()
    
    
    print("\nPlease select a model for LLM:")
    for option in llmModelOptions:
        print(f"{option}: {llmModelOptions[option]}")
    llmModel = input("\nYour choice: ").strip().upper()
    
    
    print("\nPlease select a database:")
    for option in databaseOptions:
        print(f"{option}: {databaseOptions[option]}")
    db = input("\nYour choice: ").strip().upper()
    return [embeddingModel, llmModel, db]

    
    
if __name__ == "__main__":
    main()