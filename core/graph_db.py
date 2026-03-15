from neo4j import GraphDatabase
from core.config import config
import atexit
import logging

logger = logging.getLogger(__name__)

class Neo4jClient:
    def __init__(self):
        self.uri = config.NEO4J_URI
        self.user = config.NEO4J_USER
        self.password = config.NEO4J_PASSWORD
        self.driver = None

    def connect(self):
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.user, self.password)
            )
            self.driver.verify_connectivity()
            logger.info("✅ Connected to Neo4j Knowledge Graph")
            atexit.register(self.close)
        except Exception as e:
            logger.warning(f"⚠️ Failed to connect to Neo4j: {e}")
            self.driver = None

    def close(self):
        if self.driver:
            self.driver.close()
            logger.info("Closed Neo4j connection")

    def query(self, cypher_query, parameters=None):
        if not self.driver:
            logger.warning("No active Neo4j connection.")
            return []
            
        parameters = parameters or {}
        try:
            with self.driver.session() as session:
                result = session.run(cypher_query, parameters)
                return [dict(record) for record in result]
        except Exception as e:
            logger.warning(f"Neo4j query failed: {e}")
            return []

    # Dedicated searches used by agents
    def hr_search_candidates(self, query_text):
        cypher = """
        MATCH (n)
        WHERE toLower(n.name) CONTAINS toLower($q)
        OPTIONAL MATCH (n)-[r]-(m)
        RETURN DISTINCT n.name as entity, labels(n)[0] as type, 
               collect(DISTINCT {rel: type(r), target: m.name}) as relationships
        LIMIT 5
        """
        return self.query(cypher, {"q": query_text})

    def travel_search_locations(self, query_text, limit=10):
        # Try full-text index first (if exists)
        try:
            cypher_ft = """
                CALL db.index.fulltext.queryNodes('locationFullTextIndex', $q)
                YIELD node, score
                RETURN node.id AS id, node.name AS name, node.description AS description, 
                       node.type AS type, labels(node) AS labels, score
                ORDER BY score DESC LIMIT $limit
            """
            result = self.query(cypher_ft, {"q": query_text, "limit": limit})
            if result:
                return result
        except Exception as e:
            logger.debug(f"Full-text search unavailable: {e}")

        try:
            q_lower = query_text.lower()
            cypher_fixed = """
                MATCH (n)
                WHERE any(lbl IN labels(n) WHERE lbl IN ['Location', 'City', 'Attraction', 'Hotel', 'Activity', 'Restaurant'])
                  AND (toLower(coalesce(n.name, '')) CONTAINS $q
                    OR toLower(coalesce(n.description, '')) CONTAINS $q
                    OR toLower(coalesce(n.type, '')) CONTAINS $q
                    OR toLower(coalesce(n.city, '')) CONTAINS $q
                    OR toLower(coalesce(n.region, '')) CONTAINS $q
                    OR toLower(coalesce(n.country, '')) CONTAINS $q)
                RETURN n.id AS id, n.name AS name, n.description AS description, 
                       n.type AS type, labels(n) AS labels
                LIMIT $limit
            """
            return self.query(cypher_fixed, {"q": q_lower, "limit": limit})
        except Exception as e:
            logger.error(f"Neo4j fallback search failed: {e}")
            return []

    def travel_get_relationships(self, entity_id, limit=5):
        cypher = """
            MATCH (e {id: $id})-[r]->(related)
            RETURN type(r) AS relationship, related.name AS name,
                   labels(related)[0] AS type, related.description AS description
            LIMIT $limit
        """
        return self.query(cypher, {"id": entity_id, "limit": limit})

    # ── Conflict Resolution Methods ──

    def check_superseded_by(self, doc_id: str) -> list:
        """Check if a document has been superseded by a newer one."""
        cypher = """
            MATCH (newer:Document)-[:SUPERSEDES]->(old:Document {doc_id: $doc_id})
            RETURN newer.doc_id AS newer_id, newer.name AS newer_name,
                   newer.uploaded_at AS uploaded_at, newer.version AS version
        """
        return self.query(cypher, {"doc_id": doc_id})

    def register_document(self, doc_id: str, name: str, uploaded_at: str,
                          version: str = "1", source_type: str = "database",
                          domain: str = "general"):
        """Register a Document node for provenance tracking."""
        cypher = """
            MERGE (d:Document {doc_id: $doc_id})
            SET d.name = $name,
                d.uploaded_at = $uploaded_at,
                d.version = $version,
                d.source_type = $source_type,
                d.domain = $domain
        """
        return self.query(cypher, {
            "doc_id": doc_id, "name": name, "uploaded_at": uploaded_at,
            "version": version, "source_type": source_type, "domain": domain
        })

    def mark_superseded(self, new_doc_id: str, old_doc_id: str, reason: str = ""):
        """Create a [:SUPERSEDES] relationship between two Document nodes."""
        cypher = """
            MATCH (newer:Document {doc_id: $new_id})
            MATCH (older:Document {doc_id: $old_id})
            MERGE (newer)-[r:SUPERSEDES]->(older)
            SET r.reason = $reason, r.created_at = datetime()
        """
        return self.query(cypher, {
            "new_id": new_doc_id, "old_id": old_doc_id, "reason": reason
        })


# Singleton instance
graph_db = Neo4jClient()
