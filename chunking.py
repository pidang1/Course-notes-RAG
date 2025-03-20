
# Chunk given a text into smaller pieces with specified size and overlap. Return the array of the chunks.
def chunk_text(text, chunk_size, overlap_size=0):
    
    # Split the text into tokens word by word
    tokens = text.split()
    
    # If text is shorter than chunk size, no need to split, return the chunk as is
    if len(tokens) <= chunk_size:
        return [text]
    
    chunks = []
    # start pointer to iterate over tokens
    start = 0
    
    while start < len(tokens):
        # Get the end pointer for the current chunk
        end = start + chunk_size
        
        # Get the tokens for current chunk using the pointers
        chunk_tokens = tokens[start:end]
        
        # Join tokens back into text and append to our chunks array
        chunk = ' '.join(chunk_tokens)
        chunks.append(chunk)
        
        # Move start position and offset dpeending on the overlap size
        start = end - overlap_size
    
    return chunks

# Test 
if __name__ == "__main__":
    sample_text = "Uhhh this is some sample string that has been parsed from our pdf repeated over and over" * 50
    
    # Different chunk and overlap sizes
    chunk_sizes = [200, 500, 1000]
    overlap_sizes = [0, 50, 100]
    
    # Print the chunks array for each combination 
    for chunk_size in chunk_sizes:
        for overlap_size in overlap_sizes:
            chunks = chunk_text(sample_text, chunk_size, overlap_size)
            print(f"\nChunks with size {chunk_size} and overlap {overlap_size}:")
            print(f"Number of chunks: {len(chunks)}")
