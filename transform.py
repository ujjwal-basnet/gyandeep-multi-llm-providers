import asyncio

from core.services.storage.embedding_service import EmbeddingService, index_embeddings


def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks


async def main():
    text = ""
    with open("totalBook.txt", "r") as f:
        text = f.read()

    chunks = chunk_text(text)

    embedder = EmbeddingService()
    embeddings = await embedder.get_embeddings(chunks)

    print(f"chunks: {len(chunks)}")
    print(f"embedding_dim: {len(embeddings[0]) if embeddings else 0}")

    index_embeddings(chunks, embeddings)


if __name__ == "__main__":
    asyncio.run(main())
