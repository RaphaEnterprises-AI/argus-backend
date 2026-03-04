"""
Corrective RAG (CRAG) Agent with Self-Correcting Retrieval.

Implements the Corrective RAG pattern from Yan et al. (2024) for
robust retrieval that self-corrects when initial results are poor.

RAP-254: Corrective RAG Implementation

Key Features:
- Relevance assessment of retrieved documents
- Query reformulation when results are poor
- Web search fallback for knowledge gaps
- Adaptive retrieval strategies

References:
- CRAG Paper (Yan et al. 2024): Corrective Retrieval Augmented Generation
- Self-RAG (Asai et al. 2023): Self-reflective retrieval
- Adaptive RAG (2024): Dynamic strategy selection
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import structlog

from .base import (
    AgentCapability,
    AgentResult,
    AICapability,
    AIResponse,
    BaseAgent,
)
from ..core.model_router import TaskType

logger = structlog.get_logger()


class RetrievalAction(str, Enum):
    """Actions to take based on relevance assessment."""
    CORRECT = "correct"        # Documents are relevant, use them
    INCORRECT = "incorrect"    # Documents are irrelevant, reformulate query
    AMBIGUOUS = "ambiguous"    # Partially relevant, need refinement
    WEB_SEARCH = "web_search"  # Fall back to web search


class RelevanceLevel(str, Enum):
    """Relevance levels for retrieved documents."""
    HIGHLY_RELEVANT = "highly_relevant"  # Score > 0.8
    RELEVANT = "relevant"                # Score 0.5-0.8
    MARGINALLY_RELEVANT = "marginally_relevant"  # Score 0.3-0.5
    IRRELEVANT = "irrelevant"           # Score < 0.3


@dataclass
class RetrievedDocument:
    """A retrieved document with relevance assessment."""
    id: str
    content: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    # Retrieval scores
    retrieval_score: float = 0.0  # Original retrieval score
    relevance_score: float = 0.0  # Assessed relevance
    relevance_level: RelevanceLevel = RelevanceLevel.IRRELEVANT
    relevance_reasoning: str = ""

    # Usage tracking
    used_in_response: bool = False


@dataclass
class CRAGResult:
    """Result from Corrective RAG execution."""
    query: str
    response: str
    documents_used: list[RetrievedDocument]
    retrieval_rounds: int
    actions_taken: list[str]

    # Quality metrics
    overall_relevance: float
    confidence: float
    web_search_used: bool

    # Debugging
    query_reformulations: list[str] = field(default_factory=list)
    assessment_history: list[dict[str, Any]] = field(default_factory=list)


class CorrectiveRAGAgent(BaseAgent):
    """
    Corrective RAG Agent with self-correcting retrieval.

    Unlike standard RAG that blindly uses retrieved documents, CRAG:
    1. Assesses relevance of each retrieved document
    2. Self-corrects by reformulating queries when needed
    3. Falls back to web search for knowledge gaps
    4. Only uses highly relevant documents in generation

    Usage:
        ```python
        crag = CorrectiveRAGAgent(
            retriever=hybrid_retriever,
            enable_web_fallback=True,
        )

        result = await crag.query(
            "Why is the login button test failing after the UI refactor?"
        )

        print(f"Response: {result.response}")
        print(f"Confidence: {result.confidence}")
        print(f"Documents used: {len(result.documents_used)}")
        ```
    """

    CAPABILITIES = [
        "corrective_retrieval",
        "query_reformulation",
        "relevance_assessment",
        AgentCapability.CODE_ANALYSIS,
    ]

    DEFAULT_TASK_TYPE = TaskType.SEMANTIC_UNDERSTANDING

    def __init__(
        self,
        retriever: Any = None,  # HybridRetriever or similar
        enable_web_fallback: bool = True,
        max_retrieval_rounds: int = 3,
        min_relevance_threshold: float = 0.5,
        min_documents_required: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.retriever = retriever
        self.enable_web_fallback = enable_web_fallback
        self.max_retrieval_rounds = max_retrieval_rounds
        self.min_relevance_threshold = min_relevance_threshold
        self.min_documents_required = min_documents_required

        self.log = logger.bind(agent="CorrectiveRAGAgent")

    def _get_system_prompt(self) -> str:
        return """You are a Corrective RAG agent that retrieves and synthesizes information accurately.

Your process:
1. Retrieve relevant documents for the query
2. Critically assess each document's relevance
3. If documents are poor, reformulate the query
4. Only use highly relevant information in your response
5. Clearly indicate confidence and sources

Key principles:
- Never hallucinate or make up information
- Clearly distinguish between retrieved facts and inferences
- If information is insufficient, say so
- Provide source citations for all claims"""

    async def execute(self, **kwargs) -> AgentResult:
        """Execute the CRAG agent."""
        query = kwargs.get("query", "")
        if not query:
            return AgentResult(success=False, error="No query provided")

        result = await self.query(query)
        return AgentResult(success=True, data=result)

    # =========================================================================
    # Main CRAG Flow
    # =========================================================================

    async def query(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> CRAGResult:
        """Execute Corrective RAG query with self-correction.

        Args:
            query: The user's question
            context: Optional additional context

        Returns:
            CRAGResult with response, documents, and quality metrics
        """
        self.log.info("Starting CRAG query", query=query[:100])

        actions_taken = []
        query_reformulations = []
        assessment_history = []
        all_documents: list[RetrievedDocument] = []
        current_query = query
        web_search_used = False

        for round_num in range(1, self.max_retrieval_rounds + 1):
            self.log.debug(f"Retrieval round {round_num}", query=current_query[:50])

            # Step 1: Retrieve documents
            raw_docs = await self._retrieve_documents(current_query, context)
            documents = [self._wrap_document(doc, i) for i, doc in enumerate(raw_docs)]

            # Step 2: Assess relevance of each document
            assessed_docs = await self._assess_relevance(documents, current_query)
            all_documents.extend(assessed_docs)

            # Step 3: Decide action based on assessment
            action = self._decide_action(assessed_docs)
            actions_taken.append(f"round_{round_num}:{action.value}")

            assessment_history.append({
                "round": round_num,
                "query": current_query,
                "documents_retrieved": len(assessed_docs),
                "relevant_count": sum(1 for d in assessed_docs if d.relevance_score >= self.min_relevance_threshold),
                "action": action.value,
            })

            if action == RetrievalAction.CORRECT:
                # Documents are good, proceed to generation
                self.log.info("Documents are relevant, proceeding to generation")
                break

            elif action == RetrievalAction.INCORRECT or action == RetrievalAction.AMBIGUOUS:
                # Reformulate query
                new_query = await self._reformulate_query(
                    original_query=query,
                    current_query=current_query,
                    failed_docs=assessed_docs,
                    round_num=round_num,
                )
                query_reformulations.append(new_query)
                current_query = new_query

            elif action == RetrievalAction.WEB_SEARCH and self.enable_web_fallback:
                # Fall back to web search
                web_docs = await self._web_search_fallback(current_query)
                web_search_used = True
                assessed_web = await self._assess_relevance(web_docs, current_query)
                all_documents.extend(assessed_web)

                # Check if web search helped
                if any(d.relevance_score >= self.min_relevance_threshold for d in assessed_web):
                    self.log.info("Web search provided relevant results")
                    break

        # Step 4: Filter to only relevant documents
        relevant_docs = [
            d for d in all_documents
            if d.relevance_score >= self.min_relevance_threshold
        ]

        # Deduplicate
        seen_ids = set()
        unique_docs = []
        for doc in sorted(relevant_docs, key=lambda d: d.relevance_score, reverse=True):
            if doc.id not in seen_ids:
                seen_ids.add(doc.id)
                unique_docs.append(doc)

        # Step 5: Generate response
        response, confidence = await self._generate_response(
            query=query,
            documents=unique_docs,
        )

        # Mark documents used
        for doc in unique_docs:
            doc.used_in_response = True

        # Calculate overall relevance
        overall_relevance = (
            sum(d.relevance_score for d in unique_docs) / len(unique_docs)
            if unique_docs else 0.0
        )

        return CRAGResult(
            query=query,
            response=response,
            documents_used=unique_docs,
            retrieval_rounds=len(assessment_history),
            actions_taken=actions_taken,
            overall_relevance=overall_relevance,
            confidence=confidence,
            web_search_used=web_search_used,
            query_reformulations=query_reformulations,
            assessment_history=assessment_history,
        )

    # =========================================================================
    # Retrieval
    # =========================================================================

    async def _retrieve_documents(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve documents using the configured retriever."""
        if self.retriever is None:
            # Use memory manager as fallback
            try:
                from ..orchestrator.memory_manager import get_memory_manager
                memory = get_memory_manager()
                results = await memory.search(query=query, limit=10)
                return [
                    {
                        "id": r.memory.id,
                        "content": getattr(r.memory, "summary", str(r.memory)),
                        "source": "memory",
                        "score": r.score,
                    }
                    for r in results
                ]
            except Exception:
                return []

        # Use provided retriever
        try:
            results = await self.retriever.retrieve(
                query=query,
                limit=10,
            )
            return [
                {
                    "id": getattr(r, "id", str(i)),
                    "content": getattr(r, "content", str(r)),
                    "source": getattr(r, "source", "retriever"),
                    "score": getattr(r, "score", 0.5),
                    "metadata": getattr(r, "metadata", {}),
                }
                for i, r in enumerate(results)
            ]
        except Exception as e:
            self.log.warning("Retrieval failed", error=str(e))
            return []

    def _wrap_document(self, doc: dict[str, Any], index: int) -> RetrievedDocument:
        """Wrap raw document in RetrievedDocument structure."""
        return RetrievedDocument(
            id=doc.get("id", f"doc_{index}"),
            content=doc.get("content", ""),
            source=doc.get("source", "unknown"),
            metadata=doc.get("metadata", {}),
            retrieval_score=doc.get("score", 0.0),
        )

    # =========================================================================
    # Relevance Assessment
    # =========================================================================

    async def _assess_relevance(
        self,
        documents: list[RetrievedDocument],
        query: str,
    ) -> list[RetrievedDocument]:
        """Assess relevance of each document to the query.

        This is the core of CRAG - critically assessing each document.
        """
        if not documents:
            return []

        # Batch assessment for efficiency
        assess_prompt = f"""Assess the relevance of each document to the query.

Query: {query}

Documents:
{self._format_documents_for_assessment(documents)}

For each document, provide:
1. Relevance score (0.0-1.0)
2. Relevance level (highly_relevant, relevant, marginally_relevant, irrelevant)
3. Brief reasoning

Return as JSON:
{{
    "assessments": [
        {{
            "doc_index": 0,
            "score": 0.85,
            "level": "highly_relevant",
            "reasoning": "Directly addresses the query about..."
        }}
    ]
}}

Be strict: only mark as highly_relevant if the document directly and specifically addresses the query."""

        response = await self._call_ai(
            messages=[{"role": "user", "content": assess_prompt}],
            task_type=TaskType.ERROR_CLASSIFICATION,
            json_mode=True,
            max_tokens=2000,
        )

        result = self._parse_json_response(response.content, {"assessments": []})

        # Apply assessments to documents
        for assess_data in result.get("assessments", []):
            idx = assess_data.get("doc_index", -1)
            if 0 <= idx < len(documents):
                documents[idx].relevance_score = assess_data.get("score", 0.0)
                documents[idx].relevance_level = RelevanceLevel(
                    assess_data.get("level", "irrelevant")
                )
                documents[idx].relevance_reasoning = assess_data.get("reasoning", "")

        return documents

    def _format_documents_for_assessment(self, documents: list[RetrievedDocument]) -> str:
        """Format documents for assessment prompt."""
        formatted = []
        for i, doc in enumerate(documents[:10]):  # Limit to avoid context overflow
            content_preview = doc.content[:500] + "..." if len(doc.content) > 500 else doc.content
            formatted.append(f"[{i}] Source: {doc.source}\n{content_preview}")
        return "\n\n".join(formatted)

    # =========================================================================
    # Action Decision
    # =========================================================================

    def _decide_action(
        self,
        documents: list[RetrievedDocument],
    ) -> RetrievalAction:
        """Decide what action to take based on document assessment."""
        if not documents:
            return RetrievalAction.WEB_SEARCH

        highly_relevant = sum(
            1 for d in documents
            if d.relevance_level == RelevanceLevel.HIGHLY_RELEVANT
        )
        relevant = sum(
            1 for d in documents
            if d.relevance_level in [RelevanceLevel.HIGHLY_RELEVANT, RelevanceLevel.RELEVANT]
        )
        avg_score = sum(d.relevance_score for d in documents) / len(documents)

        # Decision logic
        if highly_relevant >= self.min_documents_required:
            return RetrievalAction.CORRECT
        elif relevant >= self.min_documents_required and avg_score >= 0.6:
            return RetrievalAction.CORRECT
        elif relevant > 0:
            return RetrievalAction.AMBIGUOUS
        elif avg_score < 0.3:
            return RetrievalAction.WEB_SEARCH
        else:
            return RetrievalAction.INCORRECT

    # =========================================================================
    # Query Reformulation
    # =========================================================================

    async def _reformulate_query(
        self,
        original_query: str,
        current_query: str,
        failed_docs: list[RetrievedDocument],
        round_num: int,
    ) -> str:
        """Reformulate query to improve retrieval.

        Strategies:
        1. Decompose complex queries
        2. Add/remove specificity
        3. Use synonyms
        4. Focus on key entities
        """
        reformulation_prompt = f"""The current search query is not retrieving relevant documents.

Original query: {original_query}
Current query: {current_query}
Round: {round_num}

Retrieved documents were assessed as:
{self._format_assessment_summary(failed_docs)}

Reformulate the query to improve retrieval. Consider:
1. If too specific, broaden the search
2. If too broad, add key terms
3. Use alternative phrasings or synonyms
4. Focus on the core information need

Return ONLY the reformulated query, no explanation."""

        response = await self._call_ai(
            messages=[{"role": "user", "content": reformulation_prompt}],
            task_type=TaskType.TEXT_EXTRACTION,
            max_tokens=1024,
        )

        new_query = response.content.strip()

        # Ensure we don't just repeat the same query
        if new_query.lower() == current_query.lower():
            # Try a different approach - decomposition
            new_query = await self._decompose_query(original_query)

        self.log.debug(
            "Query reformulated",
            original=current_query[:50],
            new=new_query[:50],
        )

        return new_query

    async def _decompose_query(self, query: str) -> str:
        """Decompose a complex query into simpler sub-queries."""
        decompose_prompt = f"""Decompose this complex query into its most important sub-question:

Query: {query}

Return only the most critical sub-question that would help answer the original query."""

        response = await self._call_ai(
            messages=[{"role": "user", "content": decompose_prompt}],
            task_type=TaskType.TEXT_EXTRACTION,
            max_tokens=1024,
        )

        return response.content.strip()

    def _format_assessment_summary(self, documents: list[RetrievedDocument]) -> str:
        """Format assessment summary for reformulation prompt."""
        summary = []
        for doc in documents[:5]:
            summary.append(
                f"- Score: {doc.relevance_score:.2f}, Level: {doc.relevance_level.value}, "
                f"Reasoning: {doc.relevance_reasoning[:100]}..."
            )
        return "\n".join(summary) or "No documents retrieved"

    # =========================================================================
    # Web Search Fallback
    # =========================================================================

    async def _web_search_fallback(
        self,
        query: str,
    ) -> list[RetrievedDocument]:
        """Fall back to web search when local retrieval fails."""
        self.log.info("Falling back to web search", query=query[:50])

        # This would integrate with a web search API
        # For now, return empty (placeholder)
        try:
            # Try to use search integration if available
            from ..integrations.web_search import search_web
            results = await search_web(query, max_results=5)
            return [
                RetrievedDocument(
                    id=f"web_{i}",
                    content=r.get("snippet", r.get("content", "")),
                    source=r.get("url", "web"),
                    metadata={"title": r.get("title", ""), "url": r.get("url", "")},
                )
                for i, r in enumerate(results)
            ]
        except ImportError:
            self.log.warning("Web search not available")
            return []
        except Exception as e:
            self.log.warning("Web search failed", error=str(e))
            return []

    # =========================================================================
    # Response Generation
    # =========================================================================

    async def _generate_response(
        self,
        query: str,
        documents: list[RetrievedDocument],
    ) -> tuple[str, float]:
        """Generate response using only relevant documents.

        Returns:
            Tuple of (response, confidence)
        """
        if not documents:
            return (
                "I could not find relevant information to answer this query. "
                "Please try rephrasing or providing more context.",
                0.2,
            )

        # Build context from documents
        context_parts = []
        for i, doc in enumerate(documents[:5]):  # Use top 5
            context_parts.append(
                f"[Source {i+1}: {doc.source}]\n{doc.content}"
            )
        context = "\n\n".join(context_parts)

        generation_prompt = f"""Answer the query using ONLY the provided context.

Query: {query}

Context:
{context}

Instructions:
1. Base your answer ONLY on the provided context
2. Cite sources using [Source N] format
3. If the context doesn't fully answer the query, say so
4. Do not make up or infer information not in the context

After your answer, on a new line, provide a confidence score (0.0-1.0) based on how well the context supports your answer, formatted as: CONFIDENCE: X.XX"""

        response = await self._call_ai(
            messages=[{"role": "user", "content": generation_prompt}],
            task_type=TaskType.SEMANTIC_UNDERSTANDING,
            required_capabilities=[AICapability.REASONING],
            max_tokens=2000,
        )

        # Extract confidence from response
        content = response.content
        confidence = 0.7  # Default

        if "CONFIDENCE:" in content:
            parts = content.rsplit("CONFIDENCE:", 1)
            content = parts[0].strip()
            try:
                confidence = float(parts[1].strip()[:4])
            except (ValueError, IndexError):
                pass

        return content, confidence


# =========================================================================
# Adaptive RAG Router
# =========================================================================

class AdaptiveRAGStrategy(str, Enum):
    """RAG strategies based on query complexity."""
    NO_RETRIEVAL = "no_retrieval"      # Simple factual questions
    SINGLE_STEP = "single_step"        # Standard RAG
    CORRECTIVE = "corrective"          # CRAG with self-correction
    ITERATIVE = "iterative"            # Multi-hop reasoning
    AGENT = "agent"                    # Full agent with tools


class AdaptiveRAGRouter:
    """
    Routes queries to appropriate RAG strategy based on complexity.

    Implements Adaptive RAG pattern that analyzes query complexity
    and selects the most efficient retrieval strategy.
    """

    def __init__(self, base_agent: BaseAgent):
        self.agent = base_agent
        self.log = logger.bind(component="AdaptiveRAGRouter")

    async def route(self, query: str) -> AdaptiveRAGStrategy:
        """Determine the best RAG strategy for a query.

        Args:
            query: The user's query

        Returns:
            The recommended strategy
        """
        # Analyze query complexity
        analysis = await self._analyze_query(query)

        complexity = analysis.get("complexity", "medium")
        needs_retrieval = analysis.get("needs_retrieval", True)
        multi_hop = analysis.get("multi_hop", False)
        needs_tools = analysis.get("needs_tools", False)

        # Route based on analysis
        if not needs_retrieval:
            return AdaptiveRAGStrategy.NO_RETRIEVAL
        elif needs_tools:
            return AdaptiveRAGStrategy.AGENT
        elif multi_hop:
            return AdaptiveRAGStrategy.ITERATIVE
        elif complexity == "high":
            return AdaptiveRAGStrategy.CORRECTIVE
        else:
            return AdaptiveRAGStrategy.SINGLE_STEP

    async def _analyze_query(self, query: str) -> dict[str, Any]:
        """Analyze query to determine complexity and requirements."""
        analysis_prompt = f"""Analyze this query to determine the best retrieval strategy:

Query: {query}

Provide analysis as JSON:
{{
    "complexity": "low|medium|high",
    "needs_retrieval": true/false,
    "multi_hop": true/false,
    "needs_tools": true/false,
    "key_entities": ["entity1", "entity2"],
    "reasoning": "brief explanation"
}}

Guidelines:
- needs_retrieval: false for simple factual questions about common knowledge
- multi_hop: true if answering requires combining info from multiple sources
- needs_tools: true if it requires code execution, calculations, or API calls
- complexity: high if ambiguous, domain-specific, or requiring deep analysis"""

        response = await self.agent._call_ai(
            messages=[{"role": "user", "content": analysis_prompt}],
            task_type=TaskType.ERROR_CLASSIFICATION,
            json_mode=True,
            max_tokens=500,
        )

        return self.agent._parse_json_response(response.content, {
            "complexity": "medium",
            "needs_retrieval": True,
            "multi_hop": False,
            "needs_tools": False,
        })

    async def execute_with_strategy(
        self,
        query: str,
        strategy: AdaptiveRAGStrategy | None = None,
    ) -> dict[str, Any]:
        """Execute query with the appropriate strategy.

        Args:
            query: The user's query
            strategy: Optional explicit strategy (auto-routes if not provided)

        Returns:
            Result with response and metadata
        """
        if strategy is None:
            strategy = await self.route(query)

        self.log.info("Executing with strategy", query=query[:50], strategy=strategy.value)

        if strategy == AdaptiveRAGStrategy.NO_RETRIEVAL:
            # Direct LLM response
            response = await self.agent._call_ai(
                messages=[{"role": "user", "content": query}],
                task_type=TaskType.GENERAL,
            )
            return {"response": response.content, "strategy": strategy.value}

        elif strategy == AdaptiveRAGStrategy.CORRECTIVE:
            # Use CRAG
            crag = CorrectiveRAGAgent()
            result = await crag.query(query)
            return {
                "response": result.response,
                "strategy": strategy.value,
                "confidence": result.confidence,
                "documents_used": len(result.documents_used),
            }

        else:
            # Default to simple retrieval + generation
            # (In production, implement other strategies)
            response = await self.agent._call_ai(
                messages=[{"role": "user", "content": query}],
                task_type=TaskType.SEMANTIC_UNDERSTANDING,
            )
            return {"response": response.content, "strategy": strategy.value}
