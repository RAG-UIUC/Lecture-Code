"""
RAG UIUC Lecture: Sparse Search, Dense Search, and Hybrid Search with LLMs
===========================================================================

This lecture demonstrates three fundamental retrieval techniques used in 
Retrieval-Augmented Generation (RAG) systems:

1. Sparse Search (Keyword-based / BM25)
2. Dense Search (Semantic / Embedding-based)
3. Hybrid Search (Combination of both)

Author: RAG UIUC
Date: October 2025
"""

from typing import List, Dict, Tuple
import math
from collections import Counter
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()


# ============================================================================
# PART 1: SPARSE SEARCH (Keyword-Based / BM25)
# ============================================================================

class SparseSearch:
    """
    Sparse Search uses keyword matching and statistical methods like BM25.
    
    Advantages:
    - Fast and efficient
    - Interpretable results (exact keyword matches)
    - No need for embeddings
    - Good for exact phrase matching
    
    Disadvantages:
    - Doesn't understand semantic meaning
    - Requires exact or similar keywords
    - Poor at handling synonyms
    """
    
    def __init__(self, documents: List[str]):
        """
        Initialize sparse search with documents.
        
        Args:
            documents: List of text documents to search
        """
        self.documents = documents
        self.inverted_index = self._build_inverted_index()
        self.doc_lengths = [len(doc.split()) for doc in documents]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)
    
    def _build_inverted_index(self) -> Dict[str, List[int]]:
        """Build an inverted index: word -> [doc_ids]"""
        inverted_index = {}
        
        for doc_id, doc in enumerate(self.documents):
            words = doc.lower().split()
            for word in words:
                # Remove punctuation and normalize
                word = word.strip('.,!?;:')
                if word:
                    if word not in inverted_index:
                        inverted_index[word] = []
                    if doc_id not in inverted_index[word]:
                        inverted_index[word].append(doc_id)
        
        return inverted_index
    
    def bm25_score(self, query: str, k1: float = 1.5, b: float = 0.75) -> List[Tuple[int, float]]:
        """
        Calculate BM25 scores for each document.
        
        BM25 is a probabilistic retrieval model that considers:
        - Term frequency (TF)
        - Inverse document frequency (IDF)
        - Document length normalization
        
        Args:
            query: Search query
            k1: Parameter controlling term frequency saturation point
            b: Parameter controlling how much effect document length has
        
        Returns:
            List of (doc_id, score) tuples sorted by score
        """
        query_terms = query.lower().split()
        scores = [0.0] * len(self.documents)
        
        for term in query_terms:
            term = term.strip('.,!?;:')
            
            # Calculate IDF (Inverse Document Frequency)
            if term in self.inverted_index:
                doc_freq = len(self.inverted_index[term])
                idf = math.log(len(self.documents) - doc_freq + 0.5) / (doc_freq + 0.5)
                
                # Calculate BM25 for each document containing the term
                for doc_id in self.inverted_index[term]:
                    # Term frequency in document
                    tf = self.documents[doc_id].lower().split().count(term)
                    
                    # BM25 formula
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * (self.doc_lengths[doc_id] / self.avg_doc_length))
                    
                    scores[doc_id] += idf * (numerator / denominator)
        
        # Return sorted results
        results = [(doc_id, score) for doc_id, score in enumerate(scores) if score > 0]
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Perform sparse search on documents.
        
        Args:
            query: Search query
            top_k: Number of top results to return
        
        Returns:
            List of results with document ID, score, and content
        """
        results = self.bm25_score(query)
        
        return [
            {
                "doc_id": doc_id,
                "score": score,
                "content": self.documents[doc_id],
                "method": "BM25 (Sparse)"
            }
            for doc_id, score in results[:top_k]
        ]


# ============================================================================
# PART 2: DENSE SEARCH (Semantic / Embedding-Based)
# ============================================================================

class DenseSearch:
    """
    Dense Search uses embeddings (vector representations) to find semantically similar documents.
    
    Advantages:
    - Understands semantic meaning
    - Handles synonyms well
    - Context-aware matching
    - Better for paraphrases
    
    Disadvantages:
    - Requires embedding model
    - Slower than sparse search
    - More computationally expensive
    - API calls needed
    """
    
    def __init__(self, documents: List[str], api_key: str = "YOUR-API-KEY"):
        """
        Initialize dense search with documents.
        
        Args:
            documents: List of text documents to search
            api_key: OpenAI API key
        """
        self.documents = documents
        self.client = OpenAI(api_key="YOUR-API-KEY" or os.getenv("OPENAI_API_KEY"))
        self.embeddings = self._generate_embeddings()
    
    def _generate_embeddings(self) -> List[List[float]]:
        """
        Generate embeddings for all documents using OpenAI's API.
        
        Uses the text-embedding-3-small model which is fast and efficient.
        """
        print("Generating embeddings for documents...")
        embeddings = []
        
        for doc in self.documents:
            response = self.client.embeddings.create(
                input=doc,
                model="text-embedding-3-small"
            )
            embeddings.append(response.data[0].embedding)
        
        print(f"✅ Generated {len(embeddings)} embeddings")
        return embeddings
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a ** 2 for a in vec1))
        norm2 = math.sqrt(sum(b ** 2 for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Perform dense search using semantic similarity.
        
        Args:
            query: Search query
            top_k: Number of top results to return
        
        Returns:
            List of results with document ID, similarity score, and content
        """
        # Generate embedding for query
        query_embedding = self.client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        ).data[0].embedding
        
        # Calculate similarity scores
        scores = []
        for doc_id, doc_embedding in enumerate(self.embeddings):
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            scores.append((doc_id, similarity))
        
        # Sort by similarity score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return [
            {
                "doc_id": doc_id,
                "score": score,
                "content": self.documents[doc_id],
                "method": "Dense (Embeddings)"
            }
            for doc_id, score in scores[:top_k]
        ]


# ============================================================================
# PART 3: HYBRID SEARCH (Sparse + Dense)
# ============================================================================

class HybridSearch:
    """
    Hybrid Search combines sparse and dense search techniques.
    
    Advantages:
    - Combines strengths of both methods
    - Better recall and precision
    - Handles both keyword and semantic queries
    - More robust retrieval
    
    Strategy:
    1. Get results from sparse search
    2. Get results from dense search
    3. Combine and re-rank results
    4. Return top results
    """
    
    def __init__(self, documents: List[str], api_key: str = "YOUR-API-KEY"):
        """
        Initialize hybrid search with both sparse and dense components.
        
        Args:
            documents: List of text documents to search
            api_key: OpenAI API key
        """
        self.documents = documents
        self.sparse_search = SparseSearch(documents)
        self.dense_search = DenseSearch(documents, api_key)
    
    def _normalize_scores(self, results: List[Tuple[int, float]]) -> Dict[int, float]:
        """Normalize scores to 0-1 range."""
        if not results:
            return {}
        
        scores = [score for _, score in results]
        max_score = max(scores)
        min_score = min(scores)
        
        normalized = {}
        for doc_id, score in results:
            if max_score == min_score:
                normalized[doc_id] = 1.0
            else:
                normalized[doc_id] = (score - min_score) / (max_score - min_score)
        
        return normalized
    
    def search(self, query: str, top_k: int = 3, sparse_weight: float = 0.5, dense_weight: float = 0.5) -> List[Dict]:
        """
        Perform hybrid search combining sparse and dense results.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            sparse_weight: Weight for sparse search results (0-1)
            dense_weight: Weight for dense search results (0-1)
        
        Returns:
            List of combined and re-ranked results
        """
        # Get sparse search results
        sparse_results = self.sparse_search.bm25_score(query)
        sparse_scores = self._normalize_scores(sparse_results)
        
        # Get dense search results
        dense_results = self.dense_search.search(query, top_k=len(self.documents))
        dense_scores = {r["doc_id"]: r["score"] for r in dense_results}
        dense_scores = self._normalize_scores([(doc_id, score) for doc_id, score in dense_scores.items()])
        
        # Combine scores
        combined_scores = {}
        all_doc_ids = set(sparse_scores.keys()) | set(dense_scores.keys())
        
        for doc_id in all_doc_ids:
            sparse_score = sparse_scores.get(doc_id, 0.0)
            dense_score = dense_scores.get(doc_id, 0.0)
            
            # Weighted combination
            combined_scores[doc_id] = (sparse_weight * sparse_score + dense_weight * dense_score)
        
        # Sort and return top results
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {
                "doc_id": doc_id,
                "score": score,
                "content": self.documents[doc_id],
                "method": "Hybrid (Sparse + Dense)",
                "sparse_score": sparse_scores.get(doc_id, 0.0),
                "dense_score": dense_scores.get(doc_id, 0.0)
            }
            for doc_id, score in sorted_results[:top_k]
        ]


# ============================================================================
# PART 4: RAG SYSTEM WITH LLM AUGMENTATION
# ============================================================================

class RAGSystem:
    """
    Complete RAG (Retrieval-Augmented Generation) system combining:
    1. Retrieval (using hybrid search)
    2. Augmentation (combining with user query)
    3. Generation (using LLM to answer)
    """
    
    def __init__(self, documents: List[str], api_key: str = "YOUR-API-KEY"):
        """
        Initialize RAG system.
        
        Args:
            documents: List of documents to use as knowledge base
            api_key: OpenAI API key
        """
        self.hybrid_search = HybridSearch(documents, api_key)
        self.client = OpenAI(api_key="YOUR-API-KEY" or os.getenv("OPENAI_API_KEY"))
    
    def augment_and_generate(self, query: str, top_k: int = 3) -> Dict:
        """
        Perform RAG: Retrieve, Augment, and Generate.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
        
        Returns:
            Dictionary with retrieved documents and LLM response
        """
        # Step 1: Retrieve relevant documents
        retrieved_docs = self.hybrid_search.search(query, top_k=top_k)
        
        # Step 2: Prepare context
        context = "\n\n".join([f"Document {i+1}: {doc['content']}" 
                              for i, doc in enumerate(retrieved_docs)])
        
        # Step 3: Generate response using LLM with context
        system_prompt = """You are a helpful assistant that answers questions based on provided documents.
        
        Always cite which documents you used to answer the question.
        If the answer cannot be found in the provided documents, say so clearly."""
        
        user_message = f"""Based on the following documents, answer this question: {query}

Documents:
{context}

Please provide a clear, concise answer."""
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        return {
            "query": query,
            "retrieved_documents": retrieved_docs,
            "llm_response": response.choices[0].message.content,
            "num_documents_retrieved": len(retrieved_docs)
        }


# ============================================================================
# DEMO & TESTING
# ============================================================================

def demonstrate_retrieval_methods():
    """Demonstrate all three retrieval methods."""
    
    # Sample documents for demonstration
    documents = [
        "Machine learning is a subset of artificial intelligence that focuses on developing algorithms that can learn from data.",
        "Natural language processing is a branch of AI that deals with the interaction between computers and human language.",
        "Deep learning uses neural networks with multiple layers to process data and extract features.",
        "Computer vision enables machines to interpret and understand visual information from images and videos.",
        "Reinforcement learning is a training method where an agent learns by interacting with an environment and receiving rewards.",
        "Supervised learning involves training a model on labeled data, where the correct outputs are already known.",
        "Unsupervised learning focuses on finding patterns and relationships in data without predefined labels.",
        "Semi-supervised learning combines small amounts of labeled data with large amounts of unlabeled data to improve learning accuracy.",
        "Self-supervised learning creates its own labels from the data, reducing the need for manual annotation.",
        "Transfer learning allows models trained on one task to be reused for a different but related task.",
        "Generative models learn to create new data samples that resemble the training data.",
        "Convolutional neural networks are commonly used for image recognition and processing tasks.",
        "Recurrent neural networks are designed to handle sequential data such as time series or text.",
        "Transformers use attention mechanisms to process data in parallel, making them efficient for NLP tasks.",
        "BERT is a transformer-based model that achieves high accuracy in various natural language understanding tasks.",
        "GPT models are large-scale language models capable of generating coherent and contextually relevant text.",
        "Word embeddings map words into continuous vector spaces based on their semantic meaning.",
        "Tokenization is the process of breaking down text into smaller units such as words or subwords for analysis.",
        "Stemming and lemmatization are techniques to reduce words to their base or root form.",
        "Sentiment analysis determines the emotional tone of a text, often used in social media monitoring.",
        "Topic modeling identifies hidden themes within a collection of documents.",
        "Anomaly detection identifies rare or unusual patterns that do not conform to expected behavior.",
        "Clustering groups similar data points together based on their features.",
        "Dimensionality reduction simplifies datasets by reducing the number of input variables while preserving important information.",
        "Principal Component Analysis is a technique for reducing dimensionality by transforming data into a set of orthogonal components.",
        "t-SNE is a visualization technique that helps explore high-dimensional data in two or three dimensions.",
        "Regression analysis estimates relationships between variables to make predictions about continuous outcomes.",
        "Classification algorithms categorize data into predefined classes or labels.",
        "Decision trees make predictions by splitting data into branches based on feature conditions.",
        "Random forests combine multiple decision trees to improve prediction accuracy and prevent overfitting.",
        "Support Vector Machines classify data by finding the optimal hyperplane that separates different classes.",
        "Naive Bayes classifiers use probability theory to make predictions based on prior knowledge of conditions.",
        "K-Nearest Neighbors classifies new data points based on the majority label of nearby examples.",
        "Gradient boosting builds models sequentially, where each new model corrects errors made by previous ones.",
        "XGBoost is an optimized version of gradient boosting known for high performance in structured data problems.",
        "Hyperparameter tuning involves optimizing model settings to achieve the best performance.",
        "Cross-validation helps evaluate model performance by partitioning data into training and testing subsets multiple times.",
        "Overfitting occurs when a model learns noise from the training data and fails to generalize to new data.",
        "Underfitting happens when a model is too simple to capture the underlying structure of the data.",
        "Feature engineering is the process of creating new input features to improve model performance.",
        "Feature selection reduces the number of features to simplify models and enhance generalization.",
        "Data normalization scales input features to ensure consistency and improve model convergence.",
        "Data augmentation increases dataset size by creating modified versions of existing samples.",
        "Ethical AI focuses on building fair, transparent, and accountable artificial intelligence systems.",
        "Explainable AI aims to make machine learning models more interpretable and understandable to humans.",
        "AI bias occurs when models produce unfair outcomes due to biased training data or design choices.",
        "Edge AI deploys machine learning models on devices like smartphones or IoT sensors for real-time inference.",
        "Federated learning allows multiple devices to collaboratively train a shared model without sharing raw data.",
        "Quantum machine learning explores how quantum computing can accelerate AI algorithms.",
        "Multimodal learning integrates information from multiple data types, such as text, audio, and images."
    ]

    
    query = "What differentiats cross validation from ethical ai?"
    
    print("=" * 80)
    print("RAG LECTURE DEMONSTRATION")
    print("=" * 80)
    print(f"\nDocuments in knowledge base: {len(documents)}")
    print(f"\nQuery: '{query}'")
    print("\n" + "=" * 80)
    
    # 1. SPARSE SEARCH DEMO
    print("\n1️⃣  SPARSE SEARCH (BM25 - Keyword Based)")
    print("-" * 80)
    sparse = SparseSearch(documents)
    sparse_results = sparse.search(query, top_k=3)
    
    for i, result in enumerate(sparse_results, 1):
        print(f"\nResult {i}: (Score: {result['score']:.4f})")
        print(f"Content: {result['content'][:100]}...")
    
    # 2. DENSE SEARCH DEMO
    print("\n\n2️⃣  DENSE SEARCH (Embeddings - Semantic)")
    print("-" * 80)
    try:
        dense = DenseSearch(documents)
        dense_results = dense.search(query, top_k=3)
        
        for i, result in enumerate(dense_results, 1):
            print(f"\nResult {i}: (Score: {result['score']:.4f})")
            print(f"Content: {result['content'][:100]}...")
    except Exception as e:
        print(f"Note: Dense search requires OpenAI API key. Error: {e}")
    
    # 3. HYBRID SEARCH DEMO
    print("\n\n3️⃣  HYBRID SEARCH (Sparse + Dense Combined)")
    print("-" * 80)
    try:
        hybrid = HybridSearch(documents)
        hybrid_results = hybrid.search(query, top_k=3, sparse_weight=0.5, dense_weight=0.5)
        
        for i, result in enumerate(hybrid_results, 1):
            print(f"\nResult {i}: (Combined Score: {result['score']:.4f})")
            print(f"  Sparse Score: {result['sparse_score']:.4f}")
            print(f"  Dense Score: {result['dense_score']:.4f}")
            print(f"  Content: {result['content'][:100]}...")
    except Exception as e:
        print(f"Note: Hybrid search requires OpenAI API key. Error: {e}")
    
    # 4. FULL RAG SYSTEM DEMO
    print("\n\n4️⃣  FULL RAG SYSTEM (Retrieval + Augmentation + Generation)")
    print("-" * 80)
    try:
        rag = RAGSystem(documents)
        result = rag.augment_and_generate(query, top_k=2)
        
        print(f"\nQuery: {result['query']}")
        print(f"\nDocuments Retrieved: {result['num_documents_retrieved']}")
        print(f"\nLLM Response:\n{result['llm_response']}")
    except Exception as e:
        print(f"Note: RAG system requires OpenAI API key. Error: {e}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    demonstrate_retrieval_methods()
