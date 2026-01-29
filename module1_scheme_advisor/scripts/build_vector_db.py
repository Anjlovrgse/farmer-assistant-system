"""
Step 2: Build Vector Database using FAISS
==========================================
This script creates embeddings and builds a FAISS vector database
"""

import os
import json
import pickle
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# =============================================
# CONFIGURATION
# =============================================
TEXT_FILE = "../data/processed/schemes_text.txt"
VECTOR_DB_PATH = "../data/vector_db/"
EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"  # Fast and efficient
CHUNK_SIZE = 500  # Characters per chunk
CHUNK_OVERLAP = 50  # Overlap between chunks

# Create vector DB folder
os.makedirs(VECTOR_DB_PATH, exist_ok=True)

# =============================================
# FUNCTION: Split Text into Chunks
# =============================================
def split_text_into_chunks(text, chunk_size=500, overlap=50):
    """
    Split text into overlapping chunks
    """
    chunks = []
    words = text.split()
    
    current_chunk = []
    current_size = 0
    
    for word in words:
        current_chunk.append(word)
        current_size += len(word) + 1  # +1 for space
        
        if current_size >= chunk_size:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
            
            # Keep last few words for overlap
            overlap_words = int(overlap / 10)  # Rough estimate
            if overlap_words > 0:
                current_chunk = current_chunk[-overlap_words:]
                current_size = sum(len(w) + 1 for w in current_chunk)
            else:
                current_chunk = []
                current_size = 0
    
    # Add last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

# =============================================
# FUNCTION: Create Embeddings
# =============================================
def create_embeddings(chunks, model_name):
    """
    Create embeddings for text chunks using SentenceTransformer
    """
    print(f"\n🧠 Loading embedding model: {model_name}")
    print("   (First run will download ~90MB model - this is normal!)")
    
    try:
        model = SentenceTransformer(model_name)
        print("   ✅ Model loaded successfully")
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        print("\n💡 Try installing sentence-transformers:")
        print("   pip install sentence-transformers==2.2.2")
        return None, None
    
    print(f"\n📊 Creating embeddings for {len(chunks)} chunks...")
    print("   This may take 1-3 minutes...")
    
    try:
        embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
        print(f"   ✅ Embeddings created successfully")
        print(f"   Embedding dimension: {embeddings.shape[1]}")
        return embeddings, model
    except Exception as e:
        print(f"   ❌ Error creating embeddings: {e}")
        return None, None

# =============================================
# FUNCTION: Build FAISS Index
# =============================================
def build_faiss_index(embeddings):
    """
    Build FAISS index for fast similarity search
    """
    print(f"\n🔍 Building FAISS index...")
    
    try:
        dimension = embeddings.shape[1]
        
        # Create FAISS index (using L2 distance)
        index = faiss.IndexFlatL2(dimension)
        
        # Add embeddings to index
        index.add(embeddings.astype('float32'))
        
        print(f"   ✅ FAISS index built successfully")
        print(f"   Total vectors: {index.ntotal}")
        
        return index
    except Exception as e:
        print(f"   ❌ Error building FAISS index: {e}")
        print("\n💡 Try installing faiss-cpu:")
        print("   pip install faiss-cpu==1.7.4")
        return None

# =============================================
# MAIN: Build Vector Database
# =============================================
def main():
    print("\n" + "="*60)
    print("🔨 BUILDING VECTOR DATABASE")
    print("="*60 + "\n")
    
    # Check if text file exists
    if not os.path.exists(TEXT_FILE):
        print(f"❌ Text file not found: {TEXT_FILE}")
        print(f"\n💡 Please run extract_text.py first:")
        print(f"   cd scripts")
        print(f"   python extract_text.py")
        print(f"\n   This will create the required text file.")
        return
    
    # Read text
    print(f"📖 Reading text from: {TEXT_FILE}")
    try:
        with open(TEXT_FILE, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"   ✅ File read successfully")
        print(f"   Total characters: {len(text):,}")
    except Exception as e:
        print(f"   ❌ Error reading file: {e}")
        return
    
    if len(text) < 100:
        print(f"\n⚠️  Warning: Text file seems too small ({len(text)} characters)")
        print(f"   Make sure extract_text.py completed successfully")
        return
    
    # Split into chunks
    print(f"\n📝 Splitting text into chunks...")
    print(f"   Chunk size: {CHUNK_SIZE} characters")
    print(f"   Overlap: {CHUNK_OVERLAP} characters")
    
    chunks = split_text_into_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"   ✅ Created {len(chunks)} chunks")
    
    if len(chunks) == 0:
        print(f"\n❌ No chunks created! Check your text file.")
        return
    
    # Show sample chunk
    print(f"\n📄 Sample chunk:")
    print(f"   {chunks[0][:150]}...")
    
    # Create embeddings
    embeddings, model = create_embeddings(chunks, EMBEDDINGS_MODEL)
    
    if embeddings is None:
        print("\n❌ Failed to create embeddings. Setup incomplete.")
        return
    
    # Build FAISS index
    index = build_faiss_index(embeddings)
    
    if index is None:
        print("\n❌ Failed to build FAISS index. Setup incomplete.")
        return
    
    # Save everything
    print(f"\n💾 Saving vector database...")
    
    try:
        # Save FAISS index
        faiss_path = os.path.join(VECTOR_DB_PATH, "index.faiss")
        faiss.write_index(index, faiss_path)
        print(f"   ✅ FAISS index: {faiss_path}")
        
        # Save chunks
        chunks_path = os.path.join(VECTOR_DB_PATH, "chunks.pkl")
        with open(chunks_path, 'wb') as f:
            pickle.dump(chunks, f)
        print(f"   ✅ Chunks: {chunks_path}")
        
        # Save metadata
        metadata = {
            "num_chunks": len(chunks),
            "embedding_model": EMBEDDINGS_MODEL,
            "embedding_dimension": int(embeddings.shape[1]),
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "total_characters": len(text),
            "index_type": "IndexFlatL2"
        }
        
        metadata_path = os.path.join(VECTOR_DB_PATH, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"   ✅ Metadata: {metadata_path}")
        
    except Exception as e:
        print(f"   ❌ Error saving files: {e}")
        return
    
    # Print summary
    print("\n" + "="*60)
    print("✅ VECTOR DATABASE BUILT SUCCESSFULLY!")
    print("="*60)
    print(f"\n📊 Summary:")
    print(f"   - Text chunks: {len(chunks)}")
    print(f"   - Embedding model: {EMBEDDINGS_MODEL}")
    print(f"   - Vector dimension: {embeddings.shape[1]}")
    print(f"   - Index size: {index.ntotal} vectors")
    print(f"   - Database type: FAISS IndexFlatL2")
    
    print(f"\n📁 Files created in {VECTOR_DB_PATH}:")
    print(f"   - index.faiss ({os.path.getsize(faiss_path):,} bytes)")
    print(f"   - chunks.pkl ({os.path.getsize(chunks_path):,} bytes)")
    print(f"   - metadata.json ({os.path.getsize(metadata_path):,} bytes)")
    
    print(f"\n🚀 Next step: Test the RAG system")
    print(f"   python test_rag.py")
    print("="*60 + "\n")

# =============================================
# TEST FUNCTION
# =============================================
def test_search():
    """
    Quick test of the vector database
    """
    print("\n🧪 Running quick test...")
    
    # Load everything
    try:
        faiss_path = os.path.join(VECTOR_DB_PATH, "index.faiss")
        chunks_path = os.path.join(VECTOR_DB_PATH, "chunks.pkl")
        
        index = faiss.read_index(faiss_path)
        with open(chunks_path, 'rb') as f:
            chunks = pickle.load(f)
        
        model = SentenceTransformer(EMBEDDINGS_MODEL)
        
        # Test query
        test_query = "What is PM-KISAN scheme?"
        query_embedding = model.encode([test_query])
        
        # Search
        distances, indices = index.search(query_embedding.astype('float32'), 3)
        
        print(f"\n✅ Test successful!")
        print(f"   Query: '{test_query}'")
        print(f"   Found {len(indices[0])} relevant chunks")
        print(f"\n   Top result: {chunks[indices[0][0]][:100]}...")
        
    except Exception as e:
        print(f"\n⚠️  Test failed: {e}")
        print(f"   But vector database was created successfully!")

# =============================================
# RUN
# =============================================
if __name__ == "__main__":
    try:
        main()
        
        # Uncomment to run quick test after building
        # test_search()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print(f"\n💡 Troubleshooting:")
        print(f"   1. Make sure all dependencies are installed")
        print(f"   2. Check that extract_text.py was run successfully")
        print(f"   3. Verify text file exists and has content")