import os

import ollama
from neo4j import GraphDatabase

URI = "bolt://localhost:7687"  # Neo4jのURI
USER = "neo4j"  # ユーザー名
PASSWORD = os.getenv("NEO4J_PASSWORD", "password")  # パスワード

try:
    response = ollama.chat(
        model="mistral", messages=[{"role": "user", "content": "Hello world"}]
    )  # モデル名は適宜変更
    print("Ollama connection successful:")
    print(response["message"]["content"])
except Exception as e:
    print(f"Error connecting to Ollama: {e}")

try:
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    with driver.session() as session:
        greeting = (
            session.run(
                "CREATE (a:Greeting) SET a.message = $message RETURN a.message + ', from node ' + id(a)",
                message="Hello, Neo4j",
            )
            .single()
            .value()
        )
        print("\nNeo4j connection successful:")
        print(greeting)
    driver.close()
except Exception as e:
    print(f"Error connecting to Neo4j: {e}")
