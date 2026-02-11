"""
LangChain RAG Pipeline
Real LangChain integration with Groq (Llama 3.3 70B) for clinical decision support.
"""
import os
from typing import List, Dict, Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS as LangChainFAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from .config import settings, CLINICAL_SYSTEM_PROMPT
from .logging_config import get_logger, timed

log = get_logger("langchain_rag")

# Prompt template for clinical RAG
CLINICAL_RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", CLINICAL_SYSTEM_PROMPT),
    ("human", """Based on the following retrieved medical information, answer the clinical question.

## Retrieved Context
{context}

## Clinical Question
{question}

Provide a comprehensive, evidence-based response with clear sections for Clinical Assessment, Relevant Guidelines, and Recommendations."""),
])


def _get_llm(provider: str = "auto"):
    """
    Get an LLM instance based on available API keys.

    Priority: Groq (free) → OpenAI (paid) → None

    Args:
        provider: 'groq', 'openai', or 'auto' (try in order)

    Returns:
        LangChain LLM instance or None
    """
    groq_key = settings.groq_api_key or os.getenv("GROQ_API_KEY", "")
    openai_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY", "")

    if provider in ("groq", "auto") and groq_key:
        try:
            from langchain_groq import ChatGroq
            llm = ChatGroq(
                api_key=groq_key,
                model_name=settings.groq_model,
                temperature=0.3,
                max_tokens=1500,
            )
            log.info(f"Using Groq LLM: {settings.groq_model}")
            return llm
        except Exception as e:
            log.warning(f"Failed to initialize Groq: {e}")

    if provider in ("openai", "auto") and openai_key:
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                api_key=openai_key,
                model=settings.model_name,
                temperature=0.3,
                max_tokens=1500,
            )
            log.info(f"Using OpenAI LLM: {settings.model_name}")
            return llm
        except Exception as e:
            log.warning(f"Failed to initialize OpenAI: {e}")

    log.warning("No LLM available — retrieval-only mode")
    return None


class LangChainRAG:
    """
    LangChain-based RAG pipeline for Clinical Decision Support.

    Uses:
    - HuggingFace sentence-transformers for embeddings
    - FAISS for vector retrieval
    - Groq (Llama 3.3 70B) or OpenAI for generation
    """

    def __init__(
        self,
        llm_provider: str = "auto",
        embedding_model: str = None,
    ):
        self.embedding_model_name = embedding_model or settings.embedding_model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name
        )
        self.llm = _get_llm(llm_provider)
        self.vectorstore: Optional[LangChainFAISS] = None
        self.chain = None
        self._build_chain()

    def _build_chain(self):
        """Build the LangChain RAG chain."""
        if self.llm is None:
            self.chain = None
            return

        self.chain = (
            {
                "context": lambda x: x["context"],
                "question": lambda x: x["question"],
            }
            | CLINICAL_RAG_PROMPT
            | self.llm
            | StrOutputParser()
        )

    def load_vectorstore(self, path: str = None):
        """Load an existing FAISS index via LangChain wrapper."""
        path = path or settings.vector_store_path
        try:
            self.vectorstore = LangChainFAISS.load_local(
                path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            log.info(f"Loaded LangChain FAISS index from {path}")
        except Exception as e:
            log.warning(f"Could not load LangChain FAISS index: {e}")
            self.vectorstore = None

    def create_vectorstore(self, documents: List[Dict]):
        """
        Create a FAISS vectorstore from document dicts.

        Args:
            documents: List of dicts with 'content' and optional 'metadata'
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        lc_docs = []
        for doc in documents:
            lc_docs.append(
                Document(
                    page_content=doc["content"],
                    metadata=doc.get("metadata", {}),
                )
            )
        split_docs = splitter.split_documents(lc_docs)
        self.vectorstore = LangChainFAISS.from_documents(
            split_docs, self.embeddings
        )
        log.info(f"Created vectorstore with {len(split_docs)} chunks")

    @timed(name="langchain_rag.query")
    def query(
        self,
        question: str,
        k: int = 5,
        patient_context: str = "",
    ) -> Dict:
        """
        Run a RAG query: retrieve + generate.

        Args:
            question: Clinical question
            k: Number of documents to retrieve
            patient_context: Optional patient context to append

        Returns:
            Dict with response, sources, and metadata
        """
        # Retrieve
        retrieved = self.retrieve(question, k=k)
        context = "\n\n".join(
            f"[Source: {doc['metadata'].get('source', 'Unknown')}]\n{doc['content']}"
            for doc in retrieved
        )

        full_question = question
        if patient_context:
            full_question += f"\n\nPatient Context:\n{patient_context}"

        # Generate
        if self.chain:
            try:
                response = self.chain.invoke({
                    "context": context,
                    "question": full_question,
                })
            except Exception as e:
                log.error(f"LLM generation failed: {e}")
                response = self._fallback_response(question, context)
        else:
            response = self._fallback_response(question, context)

        return {
            "response": response,
            "retrieved_documents": retrieved,
            "query": question,
            "llm_used": self.get_llm_name(),
        }

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve relevant documents without generation."""
        if self.vectorstore is None:
            log.warning("No vectorstore loaded")
            return []

        try:
            docs_with_scores = self.vectorstore.similarity_search_with_score(
                query, k=k
            )
            results = []
            for doc, score in docs_with_scores:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(1 / (1 + score)),
                })
            return results
        except Exception as e:
            log.error(f"Retrieval failed: {e}")
            return []

    def _fallback_response(self, question: str, context: str) -> str:
        """Generate a response without LLM."""
        return (
            "## Clinical Decision Support Response\n"
            "(Note: LLM unavailable — showing retrieved information only)\n\n"
            f"### Query: {question}\n\n"
            f"### Relevant Medical Guidelines:\n{context}\n\n"
            "**Disclaimer:** Please consult with qualified healthcare professionals "
            "for medical decisions."
        )

    def get_llm_name(self) -> str:
        """Get the name of the active LLM."""
        if self.llm is None:
            return "none (retrieval-only)"
        cls = type(self.llm).__name__
        if "Groq" in cls:
            return f"Groq ({settings.groq_model})"
        if "OpenAI" in cls:
            return f"OpenAI ({settings.model_name})"
        return cls

    def is_llm_available(self) -> bool:
        """Check if an LLM is available."""
        return self.llm is not None


# Singleton
_langchain_rag: Optional[LangChainRAG] = None


def get_langchain_rag() -> LangChainRAG:
    """Get or create the LangChain RAG singleton."""
    global _langchain_rag
    if _langchain_rag is None:
        _langchain_rag = LangChainRAG()
    return _langchain_rag
