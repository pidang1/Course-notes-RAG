""" This script runs through and index the course data from the given path to your data folder,
    chunks and embed the data using various embedding models (sentence transformers/all-MiniLM-L6-v2, sentence-transformers/all-mpnet-base-v2
    and InstructorXL), and stores the data in Redis Vector DB. It will then run tests on the data and query using 2 different
    llms (llama 2 7B, Mistral 7B) to compare their performances. The resulting output will be written to a csv file. 
"""

import os
