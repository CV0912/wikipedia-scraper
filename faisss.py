import faiss
import numpy as np
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import openai

app = Flask(__name__)

# Load Faiss index and chunks
index = faiss.read_index('faiss_index.bin')
chunks = np.load('chunks.npy', allow_pickle=True)

# Load Sentence Transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Set OpenAI API key
openai.api_key = 'enter-your-key'

# Define function to retrieve top 3 relevant chunks
def retrieve_top_chunks(question, num_chunks=3):
    query_embedding = model.encode([question])[0]
    _, top_chunk_indices = index.search(np.array([query_embedding]), num_chunks)
    top_chunks = [chunks[idx] for idx in top_chunk_indices[0]]
    return top_chunks
                                                                                                                                            
# Define function to query OpenAI API
def query_openai(prompt, context):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": context + ' ' + prompt}],  # Combine question with context
        max_tokens=150,
        n=1  # Request 1 completion
    )
    answers = response.choices[0].message.content.strip()
    return answers


@app.route('/ask', methods=['POST'])
def ask_question():
    # Get the question from the request
    question = request.json['question']
    responses = []
    # Retrieve top 3 relevant chunks based on the question
    relevant_chunks = retrieve_top_chunks(question)
    for chunks in relevant_chunks:
        responses.append({'chunk ': chunks})
    answer = query_openai(question, ' '.join(relevant_chunks))    
    print(answer)
    return jsonify({'answer': answer, 'chunks': responses})

if __name__ == '__main__':
    app.run(debug=True)
