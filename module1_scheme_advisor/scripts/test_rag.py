"""
Step 3: Test RAG System
========================
Query the vector database and generate answers using Groq LLM
"""

import os
import json
import pickle
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from dotenv import load_dotenv

# Try to import Groq (optional)
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("⚠️  Groq library not installed. Install with: pip install groq")

# Load environment variables
load_dotenv()

# =============================================
# CONFIGURATION
# =============================================
VECTOR_DB_PATH = "../data/vector_db/"
EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3  # Number of relevant chunks to retrieve

# =============================================
# LOAD VECTOR DATABASE
# =============================================
def load_vector_db():
    """
    Load FAISS index and chunks
    """
    print("📂 Loading vector database...")
    
    try:
        # Load FAISS index
        faiss_path = os.path.join(VECTOR_DB_PATH, "index.faiss")
        if not os.path.exists(faiss_path):
            print(f"   ❌ FAISS index not found: {faiss_path}")
            return None, None, None
        
        index = faiss.read_index(faiss_path)
        
        # Load chunks
        chunks_path = os.path.join(VECTOR_DB_PATH, "chunks.pkl")
        if not os.path.exists(chunks_path):
            print(f"   ❌ Chunks file not found: {chunks_path}")
            return None, None, None
            
        with open(chunks_path, 'rb') as f:
            chunks = pickle.load(f)
        
        # Load metadata
        metadata_path = os.path.join(VECTOR_DB_PATH, "metadata.json")
        metadata = None
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        print(f"   ✅ Loaded {len(chunks)} chunks")
        print(f"   ✅ Index contains {index.ntotal} vectors")
        
        return index, chunks, metadata
        
    except Exception as e:
        print(f"   ❌ Error loading database: {e}")
        return None, None, None

# =============================================
# LOAD EMBEDDING MODEL
# =============================================
def load_embedding_model():
    """
    Load SentenceTransformer model
    """
    print(f"🧠 Loading embedding model: {EMBEDDINGS_MODEL}")
    
    try:
        model = SentenceTransformer(EMBEDDINGS_MODEL)
        print(f"   ✅ Model loaded successfully")
        return model
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        return None

# =============================================
# SEARCH FUNCTION
# =============================================
def search_similar_chunks(query, embedding_model, faiss_index, chunks, top_k=3):
    """
    Search for most similar chunks to query
    """
    try:
        # Create query embedding
        query_embedding = embedding_model.encode([query])
        
        # Search in FAISS
        distances, indices = faiss_index.search(query_embedding.astype('float32'), top_k)
        
        # Get relevant chunks
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(chunks):  # Validate index
                results.append({
                    "chunk": chunks[idx],
                    "distance": float(distances[0][i]),
                    "similarity_score": float(1 / (1 + distances[0][i])),  # Convert to similarity
                    "index": int(idx)
                })
        
        return results
        
    except Exception as e:
        print(f"   ❌ Error during search: {e}")
        return []

# =============================================
# GROQ LLM FUNCTION
# =============================================
def generate_answer_with_groq(query, context_chunks):
    """
    Generate answer using Groq LLM (Llama 3)
    """
    # Check if Groq is available
    if not GROQ_AVAILABLE:
        return None
    
    # Check for API key
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        print("\n⚠️  GROQ_API_KEY not found in environment!")
        print("💡 To use Groq LLM:")
        print("   1. Get free API key from: https://console.groq.com")
        print("   2. Create .env file in module3_scheme_advisor/ folder")
        print("   3. Add: GROQ_API_KEY=your_key_here")
        print("\n   Showing context without LLM generation...\n")
        return None
    
    # Initialize Groq client
    try:
        client = Groq(api_key=api_key)
    except Exception as e:
        print(f"   ❌ Error initializing Groq: {e}")
        return None
    
    # Prepare context
    context = "\n\n".join([chunk["chunk"] for chunk in context_chunks])
    
    # Create prompt
    prompt = f"""You are a helpful agricultural advisor assistant. Answer the farmer's question based ONLY on the provided government scheme information.

Context Information:
{context}

Farmer's Question: {query}

Instructions:
- Answer in simple, clear language that farmers can understand
- Only use information from the context above
- If the answer is not in the context, say "I don't have information about that in the available schemes"
- Be specific about scheme names, eligibility criteria, and benefits
- Keep the answer concise but complete
- If applicable, mention how to apply or contact information

Answer:"""
    
    # Call Groq API
    try:
        print("   🤖 Generating answer with Groq LLM...")
        
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Fast and free
            messages=[
                {"role": "system", "content": "You are a helpful agricultural advisor helping farmers understand government schemes."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Low temperature for factual responses
            max_tokens=500
        )
        
        return completion.choices[0].message.content
    
    except Exception as e:
        print(f"   ❌ Error calling Groq API: {e}")
        print(f"   💡 Check your API key and internet connection")
        return None

# =============================================
# RAG QUERY FUNCTION
# =============================================
def query_rag_system(query, embedding_model, faiss_index, chunks):
    """
    Complete RAG query: Retrieve + Generate
    """
    print(f"\n{'='*60}")
    print(f"🔍 Query: {query}")
    print(f"{'='*60}")
    
    # Step 1: Retrieve relevant chunks
    print("\n📚 Step 1: Retrieving relevant information...")
    results = search_similar_chunks(query, embedding_model, faiss_index, chunks, TOP_K)
    
    if not results:
        print("   ❌ No results found!")
        return None, []
    
    print(f"   ✅ Found {len(results)} relevant chunks\n")
    
    # Display retrieved chunks
    print("📄 Retrieved Context:")
    print("-" * 60)
    for i, result in enumerate(results, 1):
        print(f"\n[Chunk {i}] (Similarity: {result['similarity_score']:.3f})")
        print(f"{result['chunk'][:200]}{'...' if len(result['chunk']) > 200 else ''}")
    print("-" * 60)
    
    # Step 2: Generate answer
    print("\n🤖 Step 2: Generating answer...")
    answer = generate_answer_with_groq(query, results)
    
    if answer:
        print(f"\n{'='*60}")
        print("💡 GENERATED ANSWER:")
        print(f"{'='*60}")
        print(answer)
        print(f"{'='*60}\n")
    else:
        print(f"\n{'='*60}")
        print("📋 CONTEXT-ONLY MODE (No LLM):")
        print(f"{'='*60}")
        print("\nRelevant information found in documents:")
        for i, result in enumerate(results, 1):
            print(f"\n[{i}] {result['chunk']}\n")
        print(f"{'='*60}\n")
    
    return answer, results

# =============================================
# BATCH TEST FUNCTION
# =============================================
def run_test_queries(embedding_model, faiss_index, chunks):
    """
    Run a set of predefined test queries
    """
    test_queries = [
        "What is PM-KISAN scheme?",
        "How can I apply for crop insurance?",
        "What documents do I need for Kisan Credit Card?",
        "What is Soil Health Card and its benefits?",
        "Am I eligible for PM-KISAN if I'm a tenant farmer?"
    ]
    
    print("\n" + "="*60)
    print("🧪 RUNNING TEST QUERIES")
    print("="*60)
    
    results_summary = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n\n{'#'*60}")
        print(f"Test Query {i}/{len(test_queries)}")
        print(f"{'#'*60}")
        
        answer, context = query_rag_system(query, embedding_model, faiss_index, chunks)
        
        results_summary.append({
            "query": query,
            "answer_generated": answer is not None,
            "chunks_retrieved": len(context)
        })
        
        input("\n⏸️  Press Enter to continue to next query...")
    
    # Print summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    for i, result in enumerate(results_summary, 1):
        status = "✅" if result["answer_generated"] else "⚠️"
        print(f"{i}. {status} {result['query']}")
        print(f"   Chunks: {result['chunks_retrieved']}, Answer: {'Yes' if result['answer_generated'] else 'Context only'}")
    print("="*60 + "\n")

# =============================================
# MAIN: Interactive Testing
# =============================================
def main():
    print("\n" + "="*60)
    print("🤖 RAG SYSTEM - INTERACTIVE TESTING")
    print("="*60 + "\n")
    
    # Check if vector DB exists
    if not os.path.exists(VECTOR_DB_PATH):
        print(f"❌ Vector database folder not found: {VECTOR_DB_PATH}")
        print(f"\n💡 Please run build_vector_db.py first:")
        print(f"   python build_vector_db.py")
        return
    
    # Load components
    print("🔧 Initializing RAG system...\n")
    
    index, chunks, metadata = load_vector_db()
    if index is None or chunks is None:
        print("\n❌ Failed to load vector database")
        print("💡 Run build_vector_db.py first")
        return
    
    embedding_model = load_embedding_model()
    if embedding_model is None:
        print("\n❌ Failed to load embedding model")
        return
    
    # Display system info
    print("\n" + "="*60)
    print("✅ RAG SYSTEM READY!")
    print("="*60)
    print(f"📊 Database: {len(chunks)} chunks")
    print(f"🧠 Model: {EMBEDDINGS_MODEL}")
    
    if metadata:
        print(f"📐 Embedding dimension: {metadata.get('embedding_dimension', 'unknown')}")
    
    groq_status = "✅ Available" if GROQ_AVAILABLE and os.getenv("GROQ_API_KEY") else "⚠️  Not configured"
    print(f"🤖 Groq LLM: {groq_status}")
    print("="*60)
    
    # Sample queries
    sample_queries = [
        "What is PM-KISAN scheme? What are the benefits?",
        "How can I apply for crop insurance?",
        "What documents do I need for Kisan Credit Card?",
        "What is Soil Health Card and how to get it?",
        "Am I eligible for PM-KISAN if I'm a tenant farmer?",
        "Tell me about organic farming schemes",
        "What is e-NAM and how does it work?"
    ]
    
    print("\n📝 Sample queries you can try:")
    for i, q in enumerate(sample_queries, 1):
        print(f"   {i}. {q}")
    
    print("\n💡 Commands:")
    print("   - Type your question and press Enter")
    print("   - Type 'test' to run all sample queries")
    print("   - Type 'quit' or 'exit' to stop")
    
    # Interactive loop
    print("\n" + "="*60)
    
    while True:
        try:
            query = input("\n❓ Your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thank you for testing! Goodbye!\n")
                break
            
            if query.lower() == 'test':
                run_test_queries(embedding_model, index, chunks)
                continue
            
            if not query:
                print("   ⚠️  Please enter a question")
                continue
            
            # Process query
            query_rag_system(query, embedding_model, index, chunks)
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again or type 'quit' to exit\n")

# =============================================
# RUN
# =============================================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Make sure build_vector_db.py was run successfully")
        print("   2. Check that all dependencies are installed")
        print("   3. Verify .env file has GROQ_API_KEY (optional)")
        print("   4. Make sure you're in the scripts/ folder\n")