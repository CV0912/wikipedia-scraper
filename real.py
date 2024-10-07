from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import requests
import faiss
import numpy as np


def chunk_text(text, chunk_size):
    chunks = []
    words = text.split()  # Splits the text into words
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])  # Joins words to form a chunk
        chunks.append(chunk)
    return chunks
html_text=requests.get('https://en.wikipedia.org/wiki/Spider-Man').text
soup = BeautifulSoup(html_text,'lxml')

luke = soup.find('div',class_="mw-content-ltr mw-parser-output")
table = soup.find_all('table')
for t in table:
    t.decompose()    

with open('post/f1.txt','w',encoding='utf-8') as f:
    sky = luke.text
    chunks = chunk_text(sky,chunk_size=150)
    for l in range(0,len(chunks)):
        f.write(chunks[l])
        f.write("\n")
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
embeddings = model.encode(chunks)
print(embeddings.shape)#output should be (chunks,384) as the dimension of paraphrase-minilm is 384)
dimensions = embeddings.shape[1]
index = faiss.IndexFlatL2(dimensions)
index.add(embeddings)
faiss.write_index(index, 'faiss_index.bin')
np.save('chunks.npy', np.array(chunks))

