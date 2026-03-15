import asyncio
import sys
import os
import time
import logging
import requests

# Setup logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def print_banner():
    banner = """
╔══════════════════════════════════════════════════════════════╗
║             S U P E R G E N   M O D U L A R                  ║
║        Unified AI System — HR + Travel + Multi-Agent         ║
╠══════════════════════════════════════════════════════════════╣
║  Agents:                                                     ║
║    [1] 🧑‍💼 HR Assistant      — Candidate & Resume search     ║
║    [2] ✈️  Travel (Simple)   — Quick travel Q&A             ║
║    [3] 🤖 Travel (AutoGen)  — Multi-Agent itinerary team    ║
║    [0] ❌ Exit                                              ║
╚═════════════════════════════════════════════════════════════ ╝
"""
    print(banner)


def check_ollama(base_url, model, max_retries=3):
    """Check if Ollama is reachable and model is available."""
    url = f"{base_url.rstrip('/')}/api/tags"
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                models = [m.get("name", "") for m in resp.json().get("models", [])]
                # Check if our configured model exists (match with or without tag)
                model_found = any(model in m or m.startswith(model.split(":")[0]) for m in models)
                if model_found:
                    return True, f"Connected — model '{model}' ready"
                else:
                    available = ", ".join(models[:5]) if models else "(none)"
                    return False, f"Connected, but model '{model}' not found. Available: {available}"
            else:
                return False, f"HTTP {resp.status_code}"
        except requests.ConnectionError:
            if attempt < max_retries:
                print(f"     ⏳ Ollama not responding, retrying ({attempt}/{max_retries})...")
                time.sleep(2)
            else:
                return False, "Connection refused — is 'ollama serve' running?"
        except Exception as e:
            return False, str(e)
    return False, "Max retries exceeded"


def check_neo4j(uri, user, password):
    """Check if Neo4j is reachable."""
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        # Quick stat query
        with driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) as cnt")
            count = result.single()["cnt"]
        driver.close()
        return True, f"Connected — {count} nodes in graph"
    except Exception as e:
        err = str(e)
        if "refused" in err.lower() or "10061" in err:
            return False, "Connection refused — is Neo4j/Docker running?"
        return False, err[:80]


def check_chromadb(persist_dir):
    """Check if ChromaDB can be initialized locally."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=persist_dir)
        collections = client.list_collections()
        names = [c.name for c in collections] if collections else []
        return True, f"Ready — {len(names)} collection(s): {', '.join(names) if names else '(empty)'}"
    except Exception as e:
        return False, str(e)[:80]


def initialize_system():
    """Boot up all core services with thorough pre-flight checks."""
    from core.config import config

    print("=" * 60)
    print("  🔍 PRE-FLIGHT CONNECTION CHECK")
    print("=" * 60)

    # --- 1. Config ---
    print("\n  📄 Configuration (.env)")
    try:
        config.validate()
        print(f"     ✅ Valid — Model: {config.OLLAMA_MODEL}")
    except Exception as e:
        print(f"     ❌ Error: {e}")
        return False

    statuses = {}

    # --- 2. Ollama ---
    print(f"\n  🧠 Ollama LLM ({config.OLLAMA_BASE_URL})")
    ok, msg = check_ollama(config.OLLAMA_BASE_URL, config.OLLAMA_MODEL)
    statuses["ollama"] = ok
    print(f"     {'✅' if ok else '❌'} {msg}")
    if not ok:
        print("     💡 Fix: jalankan 'ollama serve' lalu 'ollama pull " + config.OLLAMA_MODEL + "'")

    # --- 3. ChromaDB ---
    print(f"\n  🗄️  ChromaDB ({config.CHROMA_PERSIST_DIR})")
    ok, msg = check_chromadb(config.CHROMA_PERSIST_DIR)
    statuses["chroma"] = ok
    print(f"     {'✅' if ok else '❌'} {msg}")

    # --- 4. Neo4j ---
    print(f"\n  🔗 Neo4j ({config.NEO4J_URI})")
    ok, msg = check_neo4j(config.NEO4J_URI, config.NEO4J_USER, config.NEO4J_PASSWORD)
    statuses["neo4j"] = ok
    print(f"     {'✅' if ok else '⚠️ '} {msg}")
    if not ok:
        print("     💡 Fix: jalankan Docker lalu 'docker start neo4j' (opsional)")

    # --- Summary ---
    print("\n" + "=" * 60)
    total = len(statuses)
    passed = sum(1 for v in statuses.values() if v)
    print(f"  📊 HASIL: {passed}/{total} services terhubung")

    if not statuses["ollama"]:
        print("\n  ⛔ Ollama WAJIB aktif untuk menjalankan SuperGen.")
        print("     Jalankan 'ollama serve' di terminal lain, lalu coba lagi.")
        return False

    if not statuses["neo4j"]:
        print("  ℹ️  Neo4j tidak aktif — sistem tetap bisa berjalan tanpa Knowledge Graph.")

    print("=" * 60)

    # --- Initialize actual modules ---
    print("\n⏳ Menginisialisasi modul internal...")

    from core.vector_db import vector_db
    vector_db.initialize()
    print("  ✅ Vector DB module loaded")

    from core.graph_db import graph_db
    graph_db.connect()
    print(f"  {'✅' if graph_db.driver else '⚠️ '} Graph DB module {'loaded' if graph_db.driver else 'skipped (offline)'}")

    print("  ✅ LLM Client ready")
    print("\n🎉 SuperGen siap digunakan!\n")
    return True


def run_hr_chat():
    """Interactive HR Assistant chat loop"""
    from agents.hr_agent import hr_agent

    print("\n" + "=" * 50)
    print("🧑‍💼 HR ASSISTANT — Recruitment & Resume Agent")
    print("=" * 50)
    print("Tanyakan apapun tentang kandidat, resume, dll.")
    print("Ketik 'back' untuk kembali ke menu.\n")

    while True:
        try:
            query = input("👤 Anda: ").strip()
            if not query:
                continue
            if query.lower() in ['back', 'kembali', 'menu']:
                break

            print("⏳ Sedang menganalisa...")
            result = hr_agent.answer_query(query)
            print(f"\n🤖 HR Agent:\n{result['answer']}")
            if result.get('sources'):
                print(f"📎 Sources: {', '.join(result['sources'])}")
            print("-" * 50 + "\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")


def run_travel_simple():
    """Interactive Travel Assistant (simple mode) chat loop"""
    from agents.travel_agent import travel_agent

    print("\n" + "=" * 50)
    print("✈️  TRAVEL ASSISTANT — Quick Q&A Mode")
    print("=" * 50)
    print("Tanyakan rekomendasi tempat, info wisata, dll.")
    print("Ketik 'back' untuk kembali ke menu.\n")

    while True:
        try:
            query = input("🌍 Anda: ").strip()
            if not query:
                continue
            if query.lower() in ['back', 'kembali', 'menu']:
                break

            print("⏳ Sedang mencari informasi...")
            result = asyncio.run(travel_agent.answer_query(query, use_autogen=False))
            print(f"\n🤖 Travel Agent [{result.get('mode', 'simple')}]:")
            print(result['answer'])
            print(f"📊 Neo4j: {result.get('neo4j_matches', 0)} matches | Vector: {result.get('vector_matches', 0)} matches")
            print("-" * 50 + "\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")


def run_travel_autogen():
    """Interactive Travel Assistant with AutoGen multi-agent team"""
    from agents.travel_agent import travel_agent

    print("\n" + "=" * 60)
    print("🤖 TRAVEL ASSISTANT — Multi-Agent Mode (AutoGen)")
    print("=" * 60)
    print("Mode ini menggunakan 4 agen AI yang berdiskusi:")
    print("  1. 📋 Planner Agent   — Menyusun rencana perjalanan")
    print("  2. 🏠 Local Agent     — Menyarankan aktivitas lokal")
    print("  3. 🗣️  Language Agent  — Tips bahasa & komunikasi")  
    print("  4. 📝 Summary Agent   — Meringkas rencana final")
    print("\n⚠️  Mode ini memakan waktu lebih lama karena 4 agen berdiskusi.")
    print("Ketik 'back' untuk kembali ke menu.\n")

    while True:
        try:
            query = input("🗺️  Anda: ").strip()
            if not query:
                continue
            if query.lower() in ['back', 'kembali', 'menu']:
                break

            print("\n🔄 Memulai sesi Multi-Agent Team...")
            print("   Planner → Local → Language → Summary Agent")
            print("   (Mohon tunggu, proses diskusi antar agen sedang berlangsung...)\n")

            result = asyncio.run(travel_agent.answer_query_autogen(query))

            print("=" * 60)
            print("📋 HASIL AKHIR (dari Travel Summary Agent):")
            print("=" * 60)
            print(result['answer'])
            print()

            # Show conversation log
            if result.get('conversation_log'):
                show_log = input("📜 Tampilkan log percakapan antar agen? (y/n): ").strip().lower()
                if show_log == 'y':
                    print("\n" + "=" * 60)
                    print("📜 LOG PERCAKAPAN MULTI-AGENT:")
                    print("=" * 60)
                    for msg in result['conversation_log']:
                        agent = msg.get('agent', 'unknown')
                        content = msg.get('content', '')
                        print(f"\n🔹 [{agent}]:")
                        print(f"   {content[:500]}{'...' if len(content) > 500 else ''}")
                    print("=" * 60)

            print(f"\n📊 Neo4j: {result.get('neo4j_matches', 0)} | Vector: {result.get('vector_matches', 0)}")
            print("-" * 60 + "\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")
            import traceback
            traceback.print_exc()


def main():
    print_banner()

    if not initialize_system():
        print("❌ Gagal inisialisasi sistem. Pastikan Ollama sedang berjalan.")
        sys.exit(1)

    while True:
        try:
            choice = input("Pilih agent [1/2/3/0]: ").strip()

            if choice == "1":
                run_hr_chat()
                print_banner()
            elif choice == "2":
                run_travel_simple()
                print_banner()
            elif choice == "3":
                run_travel_autogen()
                print_banner()
            elif choice == "0":
                print("\n👋 Terima kasih telah menggunakan SuperGen. Sampai jumpa!")
                break
            else:
                print("⚠️  Pilihan tidak valid. Masukkan 1, 2, 3, atau 0.\n")

        except KeyboardInterrupt:
            print("\n\n👋 SuperGen ditutup. Sampai jumpa!")
            break


if __name__ == "__main__":
    main()
