import asyncio
import logging
from core.vector_db import vector_db
from core.graph_db import graph_db
from core.llm_client import llm
from core.conflict_resolver import conflict_resolver

logger = logging.getLogger(__name__)


class TravelAgent:
    """Agent for Travel Itineraries and Recommendations.
    
    Supports two modes:
      - Simple mode: direct LLM call for quick queries (recommendations, factual)
      - Multi-Agent mode: AutoGen RoundRobinGroupChat for complex itinerary planning
    """

    def __init__(self):
        self.name = "Travel Assistant"

    def classify_query(self, query: str) -> str:
        q = query.lower()
        if any(word in q for word in ["itinerary", "trip", "plan", "days", "schedule"]):
            return "itinerary"
        if any(word in q for word in ["recommend", "best", "top", "where to"]):
            return "recommendation"
        if any(word in q for word in ["what is", "tell me about", "explain"]):
            return "factual"
        return "general"

    def _gather_context(self, query: str):
        """Retrieve context from both Neo4j and ChromaDB, with Conflict Resolution"""
        neo4j_results = graph_db.travel_search_locations(query, limit=10)
        chroma_results = vector_db.travel_search(query, top_k=5)
        
        # Apply Conflict Resolution on vector results
        conflict_info = ""
        if chroma_results:
            resolution = conflict_resolver.resolve(chroma_results, graph_db)
            chroma_results = resolution["results"]
            if resolution["conflicts_found"] > 0:
                conflict_info = f"\n⚡ CONFLICT RESOLUTION: {resolution['resolution_log']}\n"
                logger.info(f"Travel Conflict resolved: {resolution['resolution_log']}")
        
        context = self._build_context_str(neo4j_results, chroma_results)
        if conflict_info:
            context = conflict_info + context
        return context, neo4j_results, chroma_results

    def _build_context_str(self, neo4j_results, chroma_results):
        parts = []
        if neo4j_results:
            parts.append("=== Knowledge Graph Entities ===")
            for i, result in enumerate(neo4j_results[:5], 1):
                info = f"{i}. {result.get('name', 'Unknown')} (Type: {result.get('type', '')})"
                if result.get('description'):
                    info += f"\n   {result['description']}"
                if result.get('id'):
                    rels = graph_db.travel_get_relationships(result['id'], limit=3)
                    if rels:
                        info += "\n   Connected to: " + ", ".join(
                            [f"{r['name']} ({r['relationship']})" for r in rels]
                        )
                parts.append(info)

        if chroma_results:
            parts.append("\n=== Detailed Travel Information ===")
            for i, result in enumerate(chroma_results[:3], 1):
                if result.get("_superseded"):
                    continue  # Skip superseded documents
                title = result.get('title', result.get('metadata', {}).get('title', 'Travel Guide'))
                country = result.get('country', result.get('metadata', {}).get('country', ''))
                text = result.get('text', '')
                doc = f"{i}. From {title} ({country}):\n   {text[:400]}..."
                parts.append(doc)

        return "\n\n".join(parts)

    # ── Simple mode (direct LLM) ──
    def answer_query_simple(self, query: str):
        """Answer a query using a single direct LLM call (fast path)."""
        query_type = self.classify_query(query)
        context, neo4j_results, chroma_results = self._gather_context(query)

        system_prompts = {
            "itinerary": "You are an expert travel planner. Create detailed itineraries.",
            "recommendation": "You are a knowledgeable travel advisor. Provide specific recommendations.",
            "factual": "You are a travel expert. Provide accurate answers.",
            "general": "You are a helpful travel assistant."
        }
        sys_prompt = system_prompts.get(query_type, system_prompts["general"])

        user_prompt = f"""Based on the following information, answer this question: {query}

{context}

Instructions:
- Use specific entity names and details from the knowledge graph
- Include relevant information from the travel guides
- Be specific about locations and timing
- If creating an itinerary, organize by days
"""
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ]

        answer = llm.chat_completion(messages)

        return {
            "answer": answer,
            "mode": "simple",
            "neo4j_matches": len(neo4j_results),
            "vector_matches": len(chroma_results)
        }

    #Multi-Agent mode (AutoGen RoundRobinGroupChat)──
    async def answer_query_autogen(self, query: str):
        """Answer using AutoGen multi-agent team (deep-planning path).
        Uses 4 specialized agents working in round-robin collaboration.
        """
        from autogen_agentchat.agents import AssistantAgent
        from autogen_agentchat.conditions import TextMentionTermination
        from autogen_agentchat.teams import RoundRobinGroupChat

        # 1. Gather RAG context
        context, neo4j_results, chroma_results = self._gather_context(query)
        context_block = f"\n\n--- TRAVEL DATABASE CONTEXT ---\n{context}\n--- END CONTEXT ---" if context.strip() else ""

        # 2. Create AutoGen model client (points to local Ollama)
        model_client = llm.get_autogen_client()

        # 3. Define the 4 specialist agents
        planner_agent = AssistantAgent(
            "planner_agent",
            model_client=model_client,
            description="A helpful assistant that can plan trips.",
            system_message=(
                "You are a helpful assistant that can suggest a travel plan for a user based on their request. "
                "Use the context data provided to create accurate, location-specific plans."
                f"{context_block}"
            ),
        )

        local_agent = AssistantAgent(
            "local_agent",
            model_client=model_client,
            description="A local assistant that can suggest local activities or places to visit.",
            system_message=(
                "You are a helpful assistant that can suggest authentic and interesting local activities "
                "or places to visit for a user and can utilize any context information provided."
                f"{context_block}"
            ),
        )

        language_agent = AssistantAgent(
            "language_agent",
            model_client=model_client,
            description="A helpful assistant that can provide language tips for a given destination.",
            system_message=(
                "You are a helpful assistant that can review travel plans, providing feedback on "
                "important/critical tips about how best to address language or communication challenges "
                "for the given destination. If the plan already includes language tips, you can mention "
                "that the plan is satisfactory, with rationale."
            ),
        )

        travel_summary_agent = AssistantAgent(
            "travel_summary_agent",
            model_client=model_client,
            description="A helpful assistant that can summarize the travel plan.",
            system_message=(
                "You are a helpful assistant that can take in all of the suggestions and advice from "
                "the other agents and provide a detailed final travel plan. You must ensure that the "
                "final plan is integrated and complete. YOUR FINAL RESPONSE MUST BE THE COMPLETE PLAN. "
                "When the plan is complete and all perspectives are integrated, you can respond with TERMINATE."
            ),
        )

        # 4. Configure termination and group chat
        termination = TextMentionTermination("TERMINATE")
        group_chat = RoundRobinGroupChat(
            [planner_agent, local_agent, language_agent, travel_summary_agent],
            termination_condition=termination,
        )

        # 5. Run the team
        logger.info(f"🤖 Starting AutoGen multi-agent team for: '{query}'")

        result = await group_chat.run(task=query)

        # 6. Extract the final summary from the last message
        final_answer = ""
        agent_messages = []
        if result and result.messages:
            for msg in result.messages:
                agent_messages.append({
                    "agent": msg.source,
                    "content": msg.content
                })
            # The last message from travel_summary_agent is our final answer
            final_answer = result.messages[-1].content
            # Clean up TERMINATE marker
            final_answer = final_answer.replace("TERMINATE", "").strip()

        await model_client.close()

        return {
            "answer": final_answer,
            "mode": "multi-agent (AutoGen)",
            "agents_involved": ["planner_agent", "local_agent", "language_agent", "travel_summary_agent"],
            "conversation_log": agent_messages,
            "neo4j_matches": len(neo4j_results),
            "vector_matches": len(chroma_results)
        }

    # ── Unified entry point ──
    async def answer_query(self, query: str, use_autogen: bool = False):
        """Main entry point. Dispatches to simple or multi-agent mode."""
        if use_autogen or self.classify_query(query) == "itinerary":
            return await self.answer_query_autogen(query)
        else:
            return self.answer_query_simple(query)


travel_agent = TravelAgent()
