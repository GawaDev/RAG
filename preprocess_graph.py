import json
import os
import re  # clean_relation_type_for_cypher で使用
from typing import Any, Dict, Optional

import ollama
from neo4j import GraphDatabase

# --- ヘルパー関数 ---


def read_file_content(filepath: str) -> Optional[str]:
    """
    ファイルの内容を読み込む。JSONファイルの場合は特定のキーを結合。
    """
    try:
        if filepath.endswith(".json"):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "title" in data and "sections" in data:
                    content = f"Title: {data['title']}\n"
                    for section in data["sections"]:
                        if "heading" in section:
                            content += f"\nHeading: {section['heading']}\n"
                        if "text" in section:
                            content += section["text"] + "\n"
                    return content
                else:
                    return json.dumps(data, ensure_ascii=False, indent=2)
        elif filepath.endswith((".txt", ".md")):
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        else:
            print(f"Unsupported file type: {filepath}")
            return None
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return None


def clean_relation_type_for_cypher(relation_type: str) -> str:
    """
    Ollamaから抽出された関係性タイプをCypherのリレーションシップ名として有効な形式にクリーニングする。
    例: "<発行者>" -> "ISSUED_BY"
    """
    cleaned = relation_type.replace("<", "").replace(">", "").replace(" ", "_").upper()
    cleaned = re.sub(r"[^A-Z0-9_]", "", cleaned)
    return cleaned


def extract_entities_relations_with_ollama(
    text_chunk: str, ollama_client: ollama.Client, model_name: str
) -> Optional[Dict[str, Any]]:
    """
    Ollamaモデルを使用してテキストからエンティティと関係性を抽出する。
    抽出結果をJSON形式で整形して返す。
    """
    prompt = f"""
    以下のテキストから、主要なエンティティとその間の関係性を抽出してください。
    抽出対象のエンティティタイプは以下の通りです。
    - 人名 (PERSON)
    - 組織名 (ORGANIZATION)
    - 地名/場所 (LOCATION)
    - 概念/キーワード (CONCEPT)
    - 文書要素 (DOCUMENT_ELEMENT) - 例: ガイドライン、規則、指針、条項など

    抽出対象の関係性タイプは以下の通りです。関係性タイプは日本語の具体的な表現を<>で囲んでください。
    - <関連> (RELATES): 一般的な関連性や参照関係
    - <発行者> (ISSUED_BY): 文書や声明の発行者
    - <対象> (ADDRESSED_TO): 文書や指示の対象者や対象物
    - <言及> (MENTIONS): あるエンティティが別のエンティティに言及している
    - <定義> (DEFINES): あるエンティティが別のエンティティを定義している
    - <強調> (EMPHASIZES): あるエンティティが別のエンティティを強調している
    - <例示> (EXAMPLE_OF): あるエンティティが別のエンティティの例である

    抽出結果は、以下のJSON形式で出力してください。エンティティと関係性が見つからない場合は、空のリストを返してください。
    
    {{
      "entities": [
        {{"name": "エンティティ名", "type": "エンティティタイプ"}}
      ],
      "relations": [
        {{"entity1": "エンティティ1の名前", "relation": "<関係性タイプ>", "entity2": "エンティティ2の名前"}}
      ]
    }}
    ---
    テキスト: {text_chunk}
    出力:
    """

    try:
        response = ollama_client.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0},
        )
        # 修正：応答テキストからJSON部分を抽出するロジックを強化
        response_content = response["message"]["content"]

        # 正規表現でJSONブロックを抽出
        json_match = re.search(r"```json\s*(\{.*\})\s*```", response_content, re.DOTALL)
        if json_match:
            json_string = json_match.group(1)
        else:
            # ```json``` ブロックがない場合でも、直接JSONパースを試みる
            json_string = response_content

        print(f"Raw Ollama ERE response: {json_string}")  # デバッグ用

        # JSONパース
        parsed_data = json.loads(json_string)
        if isinstance(parsed_data, list):
            parsed_data = parsed_data[0]
            print("Note: LLM returned a list of ERE results; using the first item.")

        if not isinstance(parsed_data, dict):
            print("Unexpected ERE format: Parsed content is not a dict.")
            return None

        return parsed_data

    except json.JSONDecodeError as e:
        print(f"JSON Decode Error in ERE: {e}")
        print(f"Problematic content: {response_content}")
        return None
    except Exception as e:
        print(f"Error during ERE with Ollama: {e}")
        return None


def add_data_to_neo4j(
    extracted_data: Dict[str, Any],
    driver: GraphDatabase.driver,
    source_info: Dict[str, str],
):
    """
    抽出されたエンティティと関係性をNeo4jに登録する。
    """
    with driver.session() as session:
        # エンティティのMERGEクエリ
        entity_merge_query = """
            MERGE (e:Entity {name: $name})
            ON CREATE SET
                e.type = $type,
                e.source_file = $source_file,
                e.creation_time = timestamp()
            ON MATCH SET
                e.type = CASE WHEN e.type IS NULL THEN $type ELSE e.type END,
                e.source_file = CASE WHEN e.source_file IS NULL THEN $source_file ELSE e.source_file END
            RETURN e
        """

        # 関係性のMERGEクエリテンプレート
        # リレーションシップタイプは動的に構築
        relationship_merge_query_template = """
            MATCH (e1:Entity {{name: $entity1_name}})
            MATCH (e2:Entity {{name: $entity2_name}})
            MERGE (e1)-[r:`{relation_type_cleaned}`]->(e2)
            ON CREATE SET
                r.source_file = $source_file,
                r.creation_time = timestamp()
            RETURN r
        """

        # エンティティの登録
        if "entities" in extracted_data and extracted_data["entities"]:
            for entity in extracted_data["entities"]:
                try:
                    # name と type はOllamaからの抽出結果、source_file は引数から
                    session.write_transaction(
                        lambda tx, name, type, source_file: tx.run(
                            entity_merge_query,
                            name=name,
                            type=type,
                            source_file=source_file,
                        ).single(),
                        entity.get("name"),
                        entity.get("type"),
                        source_info.get("source_file"),
                    )
                    # print(f"  Merged entity: {entity.get('name')} ({entity.get('type')})")
                except Exception as e:
                    print(f"Error merging entity {entity.get('name')}: {e}")

        # 関係性の登録
        if "relations" in extracted_data and extracted_data["relations"]:
            for relation in extracted_data["relations"]:
                try:
                    entity1_name = relation.get("entity1")
                    relation_type = relation.get("relation")
                    entity2_name = relation.get("entity2")

                    if entity1_name and relation_type and entity2_name:
                        cleaned_type = clean_relation_type_for_cypher(relation_type)

                        # 動的な関係性タイプをクエリに埋め込む
                        full_query = relationship_merge_query_template.format(
                            relation_type_cleaned=cleaned_type
                        )

                        session.write_transaction(
                            lambda tx, e1_name, e2_name, source_file: tx.run(
                                full_query,
                                entity1_name=e1_name,
                                entity2_name=e2_name,
                                source_file=source_file,
                            ).single(),
                            entity1_name,
                            entity2_name,
                            source_info.get("source_file"),
                        )
                        # print(f"  Merged relation: ({entity1_name})-[{cleaned_type}]->({entity2_name})")
                    else:
                        print(f"  Skipping incomplete relation: {relation}")
                except Exception as e:
                    print(f"Error merging relation {relation}: {e}")


# --- Graphデータ前処理のメイン関数 ---
def preprocess_for_graph(
    data_dir: str,
    ere_model: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    ollama_client: ollama.Client,
) -> bool:
    """
    指定されたディレクトリからファイルを読み込み、エンティティと関係性を抽出し、Neo4jに登録する。
    成功した場合はTrue、失敗した場合はFalseを返す。
    """
    driver = None
    try:
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        driver.verify_connectivity()
        print("Neo4j connection successful for graph preprocessing.")
    except Exception as e:
        print(f"Error connecting to Neo4j for graph preprocessing: {e}")
        return False

    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found at {data_dir}")
        if driver:
            driver.close()
        return False

    print(
        f"Starting GraphRAG preprocessing from {data_dir} using model '{ere_model}'..."
    )

    success_count = 0
    total_files = 0
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        if os.path.isfile(filepath):
            total_files += 1
            print(f"Processing file for GraphRAG: {filename}")
            content = read_file_content(filepath)
            if content:
                # ファイル全体をEREにかける（必要に応じてチャンク化）
                # 現在はファイル全体を1チャンクとして扱っています
                extracted_data = extract_entities_relations_with_ollama(
                    content, ollama_client, ere_model
                )

                if extracted_data and (
                    extracted_data.get("entities") or extracted_data.get("relations")
                ):
                    source_info = {
                        "source_file": filename,
                        "chunk_id": 0,  # ファイル全体を一つのチャンクとして扱う場合
                    }
                    try:
                        add_data_to_neo4j(extracted_data, driver, source_info)
                        print(
                            f"  Successfully added graph data for {filename} to Neo4j."
                        )
                        success_count += 1
                    except Exception as e:
                        print(f"  Error adding graph data for {filename} to Neo4j: {e}")
                else:
                    print(
                        f"  No entities or relations extracted or parsed correctly from {filename}."
                    )
            else:
                print(f"  Skipping empty or unreadable file for GraphRAG: {filename}")

    print(
        f"--- Graph Preprocessing Complete. Successfully processed {success_count} of {total_files} file(s). ---"
    )

    if driver:
        driver.close()

    # 少なくとも1つのファイルが成功裏に処理された場合に True を返す
    return success_count > 0


# --- スクリプトとして実行された場合の処理 (app.pyからは直接呼ばれない) ---
if __name__ == "__main__":
    # このブロック内の処理は、ファイルを直接実行したときのテストや初回処理用です。
    # app.py は上記の preprocess_for_graph 関数を直接呼び出します。

    # テスト用の設定（環境変数から読み込むか、直接指定）
    OLLAMA_HOST_TEST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ERE_MODEL_TEST = "mistral"
    DATA_DIR_TEST = "./data/raw"
    NEO4J_URI_TEST = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER_TEST = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD_TEST = os.getenv(
        "NEO4J_PASSWORD", "password"
    )  # YOUR_NEO4J_PASSWORD_HERE

    try:
        test_ollama_client = ollama.Client(host=OLLAMA_HOST_TEST)
        test_ollama_client.version()  # 接続確認
        print(
            f"Test Ollama client connected successfully. Version: {test_ollama_client.version()}"
        )
    except Exception as e:
        print(
            f"Test Ollama client connection failed: {e}. Ensure Ollama is running at {OLLAMA_HOST_TEST}"
        )
        exit()

    print("\nAttempting Graph preprocessing test run...")
    success = preprocess_for_graph(
        DATA_DIR_TEST,
        ERE_MODEL_TEST,
        NEO4J_URI_TEST,
        NEO4J_USER_TEST,
        NEO4J_PASSWORD_TEST,
        test_ollama_client,
    )

    if success:
        print(
            "\nGraph preprocessing test run completed SUCCESSFULLY. Check your Neo4j Browser."
        )
    else:
        print("\nGraph preprocessing test run FAILED. Check logs for errors.")
