"""
Flask API for Government Scheme Advisor
========================================
RESTful API for RAG-based scheme recommendations
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import pickle
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv
import sys
import sqlite3
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

# Try to import Groq
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("⚠️  Groq not installed. Install with: pip install groq")

# Load environment variables
load_dotenv()

# =============================================
# INITIALIZE FLASK APP
# =============================================
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# =============================================
# CONFIGURATION
# =============================================
# Base directory of this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_PATH = os.path.join(BASE_DIR, "../data/vector_db/")
EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3

# =============================================
# GLOBAL VARIABLES
# =============================================
embedding_model = None
faiss_index = None
chunks = None
metadata = None
groq_client = None
initialization_errors = []

# =============================================
# LOAD COMPONENTS ON STARTUP
# =============================================
def initialize_rag_system():
    """
    Load all RAG components
    """
    global embedding_model, faiss_index, chunks, metadata, groq_client, initialization_errors
    
    print("\n" + "="*60)
    print("🚀 INITIALIZING RAG SYSTEM")
    print("="*60 + "\n")
    
    initialization_errors = []
    
    try:
        # 1. Load FAISS index
        print("📂 Loading FAISS index...")
        faiss_path = os.path.join(VECTOR_DB_PATH, "index.faiss")
        
        if not os.path.exists(faiss_path):
            error_msg = f"FAISS index not found: {faiss_path}"
            print(f"   ❌ {error_msg}")
            initialization_errors.append(error_msg)
            return False
        
        faiss_index = faiss.read_index(faiss_path)
        print(f"   ✅ FAISS index loaded ({faiss_index.ntotal} vectors)")
        
        # 2. Load chunks
        print("📄 Loading text chunks...")
        chunks_path = os.path.join(VECTOR_DB_PATH, "chunks.pkl")
        
        if not os.path.exists(chunks_path):
            error_msg = f"Chunks file not found: {chunks_path}"
            print(f"   ❌ {error_msg}")
            initialization_errors.append(error_msg)
            return False
        
        with open(chunks_path, 'rb') as f:
            chunks = pickle.load(f)
        print(f"   ✅ Chunks loaded ({len(chunks)} chunks)")
        
        # 3. Load metadata
        print("📋 Loading metadata...")
        metadata_path = os.path.join(VECTOR_DB_PATH, "metadata.json")
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"   ✅ Metadata loaded")
        else:
            print(f"   ⚠️  Metadata not found (optional)")
        
        # 4. Load embedding model
        print(f"🧠 Loading embedding model: {EMBEDDINGS_MODEL}...")
        embedding_model = SentenceTransformer(EMBEDDINGS_MODEL)
        print(f"   ✅ Embedding model loaded")
        
        # 5. Initialize Groq client (optional)
        if GROQ_AVAILABLE:
            api_key = os.getenv("GROQ_API_KEY")
            if api_key:
                try:
                    groq_client = Groq(api_key=api_key)
                    print("   ✅ Groq LLM initialized")
                except Exception as e:
                    print(f"   ⚠️  Groq initialization failed: {e}")
                    print("   API will work in context-only mode")
            else:
                print("   ⚠️  GROQ_API_KEY not found in .env")
                print("   API will work in context-only mode")
        else:
            print("   ⚠️  Groq library not available")
            print("   Install with: pip install groq")
        
        print("\n" + "="*60)
        print("✅ RAG SYSTEM INITIALIZED SUCCESSFULLY!")
        print("="*60 + "\n")
        
        return True
        
    except Exception as e:
        error_msg = f"Initialization error: {str(e)}"
        print(f"\n❌ {error_msg}\n")
        initialization_errors.append(error_msg)
        return False

def init_db():
    """Initialize SQLite database for search history and users"""
    try:
        os.makedirs(os.path.dirname(VECTOR_DB_PATH), exist_ok=True)
        db_path = os.path.join(os.path.dirname(VECTOR_DB_PATH), 'search_history.db')
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        # Create users table
        c.execute('''CREATE TABLE IF NOT EXISTS users
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      username TEXT UNIQUE NOT NULL,
                      password_hash TEXT NOT NULL,
                      created_at TEXT)''')
                      
        # Create history table
        c.execute('''CREATE TABLE IF NOT EXISTS history
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      timestamp TEXT,
                      user_id INTEGER,
                      question TEXT,
                      answer TEXT,
                      llm_used BOOLEAN,
                      FOREIGN KEY(user_id) REFERENCES users(id))''')
                      
        conn.commit()
        conn.close()
        print("   ✅ SQLite database (users & history) initialized")
    except Exception as e:
        print(f"   ⚠️  Could not initialize database: {e}")

def save_to_history(user_id, question, answer, llm_used):
    """Save a search query and answer to the database"""
    try:
        db_path = os.path.join(os.path.dirname(VECTOR_DB_PATH), 'search_history.db')
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("INSERT INTO history (timestamp, user_id, question, answer, llm_used) VALUES (?, ?, ?, ?, ?)",
                  (datetime.now().isoformat(), user_id, question, answer, llm_used))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error saving history: {e}")

# Initialize on startup
init_db()
rag_ready = initialize_rag_system()

# =============================================
# RAG FUNCTIONS
# =============================================
def search_chunks(query, top_k=3):
    """
    Search for relevant chunks
    """
    try:
        query_embedding = embedding_model.encode([query])
        distances, indices = faiss_index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(chunks):
                results.append({
                    "chunk": chunks[idx],
                    "score": float(1 / (1 + distances[0][i])),  # Convert to similarity
                    "distance": float(distances[0][i]),
                    "index": int(idx)
                })
        
        return results
    except Exception as e:
        print(f"Search error: {e}")
        return []

def generate_answer(query, context_chunks):
    """
    Generate answer using Groq LLM
    """
    if not groq_client:
        # Return context without LLM
        return {
            "answer": "LLM not available. Here are relevant excerpts from government schemes:",
            "context": [c["chunk"][:300] + "..." if len(c["chunk"]) > 300 else c["chunk"] for c in context_chunks],
            "llm_used": False
        }
    
    # Prepare context
    context = "\n\n".join([c["chunk"] for c in context_chunks])
    
    # Create prompt
    prompt = f"""You are a helpful agricultural advisor. Answer the farmer's question based ONLY on the context below.

Context from Government Schemes:
{context}

Farmer's Question: {query}

Instructions:
- Answer in simple, clear language
- Only use information from the context
- If not in context, say "I don't have information about that in the available schemes"
- Be specific about scheme names, eligibility, benefits
- Mention how to apply if relevant
- Keep answer concise but complete

Answer:"""
    
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful agricultural advisor helping farmers understand government schemes."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        return {
            "answer": completion.choices[0].message.content,
            "context": [c["chunk"][:300] + "..." if len(c["chunk"]) > 300 else c["chunk"] for c in context_chunks],
            "llm_used": True
        }
    
    except Exception as e:
        return {
            "answer": f"Error generating answer: {str(e)}. Here is the relevant context:",
            "context": [c["chunk"][:300] + "..." if len(c["chunk"]) > 300 else c["chunk"] for c in context_chunks],
            "llm_used": False,
            "error": str(e)
        }

# =============================================
# API ROUTES
# =============================================

@app.route('/')
def home():
    """
    Home endpoint - API info
    """
    return jsonify({
        "message": "🌾 Government Scheme Advisor API",
        "version": "1.0",
        "status": "active" if rag_ready else "initialization_failed",
        "components": {
            "vector_db": faiss_index is not None,
            "embedding_model": embedding_model is not None,
            "chunks_loaded": chunks is not None,
            "llm_available": groq_client is not None
        },
        "endpoints": {
            "/": "GET - API information",
            "/health": "GET - Health check",
            "/register": "POST - Create a new user account",
            "/login": "POST - Login via username and password",
            "/ask": "POST - Ask a question (body: {\"question\": \"...\", \"user_id\": 1})",
            "/search": "POST - Search only (body: {\"query\": \"...\"})",
            "/history": "GET - Get search history logs",
            "/stats": "GET - Database statistics"
        },
        "documentation": "Send POST to /ask with JSON: {\"question\": \"What is PM-KISAN?\"}"
    }), 200

@app.route('/health')
def health():
    """
    Health check endpoint
    """
    health_status = {
        "status": "healthy" if rag_ready else "unhealthy",
        "components": {
            "vector_db": faiss_index is not None,
            "embedding_model": embedding_model is not None,
            "chunks_loaded": chunks is not None,
            "llm_available": groq_client is not None
        },
        "database_size": len(chunks) if chunks else 0,
        "vector_count": faiss_index.ntotal if faiss_index else 0
    }
    
    if initialization_errors:
        health_status["errors"] = initialization_errors
    
    status_code = 200 if rag_ready else 503
    return jsonify(health_status), status_code

@app.route('/register', methods=['POST'])
def register():
    """Register a new user account"""
    data = request.get_json()
    if not data or 'username' not in data or 'password' not in data:
        return jsonify({"success": False, "error": "Missing username or password"}), 400
        
    username = data['username'].strip()
    password = data['password'].strip()
    
    if len(username) < 3 or len(password) < 4:
        return jsonify({"success": False, "error": "Username must be > 3 chars and password > 4 chars"}), 400
        
    try:
        db_path = os.path.join(os.path.dirname(VECTOR_DB_PATH), 'search_history.db')
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        # Check if username exists
        c.execute("SELECT id FROM users WHERE username = ?", (username,))
        if c.fetchone():
            return jsonify({"success": False, "error": "Username already taken"}), 409
            
        hashed_pw = generate_password_hash(password)
        c.execute("INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                  (username, hashed_pw, datetime.now().isoformat()))
        conn.commit()
        user_id = c.lastrowid
        conn.close()
        
        return jsonify({
            "success": True, 
            "message": "Registration successful",
            "user": {"id": user_id, "username": username}
        }), 201
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/login', methods=['POST'])
def login():
    """Authenticate a user"""
    data = request.get_json()
    if not data or 'username' not in data or 'password' not in data:
        return jsonify({"success": False, "error": "Missing username or password"}), 400
        
    username = data['username'].strip()
    password = data['password'].strip()
    
    try:
        db_path = os.path.join(os.path.dirname(VECTOR_DB_PATH), 'search_history.db')
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        c.execute("SELECT id, username, password_hash FROM users WHERE username = ?", (username,))
        user = c.fetchone()
        conn.close()
        
        if user and check_password_hash(user['password_hash'], password):
            return jsonify({
                "success": True, 
                "message": "Login successful",
                "user": {"id": user['id'], "username": user['username']}
            }), 200
        else:
            return jsonify({"success": False, "error": "Invalid username or password"}), 401
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask():
    """
    Main RAG endpoint - Ask a question
    Expects: {"question": "your question here"}
    """
    if not rag_ready:
        return jsonify({
            "success": False,
            "error": "RAG system not initialized",
            "details": initialization_errors
        }), 503
    
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No JSON data provided"
            }), 400
        
        if 'question' not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'question' field in request",
                "example": {"question": "What is PM-KISAN scheme?"}
            }), 400
        
        question = data['question'].strip()
        user_id = data.get('user_id') # Optional user ID
        
        if not question:
            return jsonify({
                "success": False,
                "error": "Question cannot be empty"
            }), 400
        
        # Search relevant chunks
        relevant_chunks = search_chunks(question, TOP_K)
        
        if not relevant_chunks:
            return jsonify({
                "success": True,
                "question": question,
                "answer": "I couldn't find relevant information in the database. Please try rephrasing your question.",
                "sources": [],
                "llm_used": False,
                "num_sources": 0
            }), 200
        
        # Generate answer
        result = generate_answer(question, relevant_chunks)
        
        # Save to history database
        save_to_history(user_id, question, result["answer"], result["llm_used"])
        
        # Prepare response
        response = {
            "success": True,
            "question": question,
            "answer": result["answer"],
            "sources": result["context"],
            "llm_used": result["llm_used"],
            "num_sources": len(relevant_chunks),
            "similarity_scores": [chunk["score"] for chunk in relevant_chunks]
        }
        
        if "error" in result:
            response["llm_error"] = result["error"]
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Internal server error: {str(e)}"
        }), 500

@app.route('/search', methods=['POST'])
def search():
    """
    Search endpoint - Returns relevant chunks without LLM
    Expects: {"query": "search term", "top_k": 3}
    """
    if not rag_ready:
        return jsonify({
            "success": False,
            "error": "RAG system not initialized"
        }), 503
    
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'query' field in request"
            }), 400
        
        query = data['query'].strip()
        top_k = data.get('top_k', TOP_K)
        
        # Search
        results = search_chunks(query, top_k)
        
        return jsonify({
            "success": True,
            "query": query,
            "results": results,
            "count": len(results)
        }), 200
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/history', methods=['GET'])
def get_history():
    """
    Get backend search history from database
    """
    try:
        db_path = os.path.join(os.path.dirname(VECTOR_DB_PATH), 'search_history.db')
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        # Get optional limit, default to 50
        limit = request.args.get('limit', 50, type=int)
        
        c.execute("SELECT * FROM history ORDER BY timestamp DESC LIMIT ?", (limit,))
        rows = c.fetchall()
        
        history = []
        for row in rows:
            history.append({
                "id": row["id"],
                "timestamp": row["timestamp"],
                "question": row["question"],
                "answer": row["answer"],
                "llm_used": bool(row["llm_used"])
            })
            
        conn.close()
        
        return jsonify({
            "success": True,
            "count": len(history),
            "history": history
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/stats')
def stats():
    """
    Get database statistics
    """
    if not rag_ready:
        return jsonify({
            "success": False,
            "error": "RAG system not initialized"
        }), 503
    
    stats_data = {
        "success": True,
        "database": {
            "total_chunks": len(chunks) if chunks else 0,
            "vector_dimension": metadata.get('embedding_dimension') if metadata else None,
            "chunk_size": metadata.get('chunk_size') if metadata else None,
            "embedding_model": EMBEDDINGS_MODEL
        },
        "api": {
            "llm_available": groq_client is not None,
            "top_k_default": TOP_K
        }
    }
    
    if metadata:
        stats_data["metadata"] = metadata
    
    return jsonify(stats_data), 200

# =============================================
# ERROR HANDLERS
# =============================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "success": False,
        "error": "Endpoint not found",
        "available_endpoints": ["/", "/health", "/ask", "/search", "/stats"]
    }), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        "success": False,
        "error": "Internal server error"
    }), 500

# =============================================
# RUN APP
# =============================================
if __name__ == '__main__':
    if not rag_ready:
        print("\n" + "="*60)
        print("⚠️  RAG SYSTEM NOT READY!")
        print("="*60)
        print("\n💡 Please complete these steps first:\n")
        print("1. Run extract_text.py to extract text from PDFs")
        print("2. Run build_vector_db.py to build vector database")
        print("3. (Optional) Add GROQ_API_KEY to .env file\n")
        print("Errors encountered:")
        for error in initialization_errors:
            print(f"   ❌ {error}")
        print("\n" + "="*60)
        print("\n⚠️  API will start but will return errors until fixed!\n")
    else:
        print("\n" + "="*60)
        print("💬 GOVERNMENT SCHEME ADVISOR API")
        print("="*60)
        print(f"\n📊 Database Statistics:")
        print(f"   - Chunks: {len(chunks)}")
        print(f"   - Vectors: {faiss_index.ntotal}")
        print(f"   - Model: {EMBEDDINGS_MODEL}")
        print(f"   - LLM: {'Groq Llama 3.1 ✅' if groq_client else 'Not available ⚠️'}")
        print(f"\n🌐 API Endpoints:")
        print(f"   - GET  /         - API info")
        print(f"   - GET  /health   - Health check")
        print(f"   - POST /register - Create user")
        print(f"   - POST /login    - Authenticate user")
        print(f"   - POST /ask      - Ask questions")
        print(f"   - POST /search   - Search only")
        print(f"   - GET  /history  - View search history")
        print(f"   - GET  /stats    - Statistics")
        print(f"\n🚀 Server running on: http://127.0.0.1:5000")
        print(f"   Press Ctrl+C to stop")
        print("="*60 + "\n")
    
    # Run Flask app
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        use_reloader=False  # Prevent double initialization
    )