import fitz, os, time
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv(".env.local")

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(os.environ["PINECONE_INDEX"])

docs_dir = "./documents"
files = [f for f in os.listdir(docs_dir) if f.endswith(".pdf")]
print(f"Found {len(files)} PDFs")

total = 0
for file in files:
    print(f"Processing {file}...")
    doc = fitz.open(os.path.join(docs_dir, file))
    text = " ".join(page.get_text() for page in doc)
    words = text.split()
    chunks = [" ".join(words[i:i+400]) for i in range(0, len(words), 350) if len(" ".join(words[i:i+400])) > 100]
    name = file.replace(".pdf", "")
    for i in range(0, len(chunks), 20):
        batch = chunks[i:i+20]
        embeddings = pc.inference.embed("llama-text-embed-v2", batch, {"input_type": "passage", "truncate": "END"})
        records = [{"id": f"{name}-{i+j}", "values": list(embeddings[j].values), "metadata": {"text": batch[j], "source": name}} for j in range(len(batch))]
        index.upsert(vectors=records)
        print(f"  Batch {i//20+1}/{(len(chunks)+19)//20} done")
        time.sleep(0.5)
    total += len(chunks)
    print(f"  ✅ {file} — {len(chunks)} chunks")

print(f"\n✅ Done! {total} total chunks uploaded.")