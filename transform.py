from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

chunks = [] 
text = ""
with open("totalBook.txt", "r") as f:
    text = f.read()

chunks = chunk_text(text)

embeddings = model.encode(chunks)

print(embeddings)
print(embeddings.shape)