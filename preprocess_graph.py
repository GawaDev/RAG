# preprocess_graph.py

import json
import os
from typing import Any, Dict, List, Optional

import ollama
from neo4j import GraphDatabase

# --- 設定 ---
# 事前入力データが配置されているディレクトリ (preprocess_data_rag.py と同じ)
DATA_DIR = "data/raw"
# エンティティ・関係性抽出に使用するOllamaモデル名 (LLMモデル, 例: mistral, llama3)
# 高度な指示理解ができるモデルが適しています
ERE_MODEL = "mistral"
# Neo4j接続情報 (check_services.py と同じ)
URI = "bolt://localhost:7687"  # または "neo4j://localhost:7687"
USER = "neo4j"
PASSWORD = os.getenv(
    "NEO4J_PASSWORD", "HIROTAHIROTA"
)  # 環境変数から取得、なければデフォルト

# --- ヘルパー関数 ---


def read_file_content(filepath: str) -> str:
    """
    指定されたファイルのコンテンツを読み込む (preprocess_data_rag.py と同じ)
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except UnicodeDecodeError:
        try:
            with open(filepath, "r", encoding="shift_jis") as f:
                content = f.read()
            return content
        except Exception as e:
            print(
                f"Warning: Could not read file {filepath} with utf-8 or shift_jis: {e}"
            )
            return ""
    except Exception as e:
        print(f"Warning: Could not read file {filepath}: {e}")
        return ""


# シンプルなチャンク分割関数 (EREのために使う場合は、少し大きめのチャンクでも良い)
# preprocess_data_rag.py と同じ関数を再利用またはインポートしても良い
def simple_chunk_text_for_ere(
    text: str, chunk_size: int = 1500, chunk_overlap: int = 100
) -> List[str]:
    """
    ERE用のチャンク分割 (デフォルトサイズを大きめに設定)
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start : min(end, len(text))]
        chunks.append(chunk)
        if end >= len(text):
            break
        start += chunk_size - chunk_overlap
        if chunk_overlap >= chunk_size and start < len(text):
            pass
    return chunks


def extract_entities_relations_with_ollama(
    text: str, ollama_client: ollama.Client, model_name: str
) -> Optional[Dict[str, List[Dict[str, str]]]]:
    # Ollamaへの指示
    prompt = f"""
以下のテキストから主要なエンティティとその間の関係性を抽出してください。

抽出対象のエンティティタイプ：
- 人名 (PERSON)
- 組織名 (ORGANIZATION)
- 場所 (LOCATION)
- 概念/キーワード (CONCEPT)
- 文書要素 (DOCUMENT_ELEMENT)

抽出対象の関係性タイプ：
- 関連 (RELATES) - 単純な関連性
- 発行者 (ISSUED_BY) - 文書等の発行元
- 対象 (ADDRESSED_TO) - 文書等の対象者
- 言及 (MENTIONS) - あるエンティティが別のエンティティについて言及
- 定義 (DEFINES) - ある概念が別の概念を定義
- 強調 (EMPHASIZES) - ある文書や概念が別の概念や点を強調
- 例示 (EXAMPLE_OF) - あるエンティティが別の概念の例である

抽出結果は、以下のJSON形式で出力してください。**JSON以外の情報や、```json のようなマークアップは一切含めないでください。** JSONオブジェクトのみを出力してください。
{{
  "entities": [
    {{"name": "エンティティ名", "type": "エンティティタイプ"}},
    ...
  ],
  "relations": [
    {{"entity1": "エンティティ名1", "relation": "関係性タイプ", "entity2": "エンティティ名2"}},
    ...
  ]
}}

エンティティ名や関係性タイプは、元のテキストに含まれる表現に合わせてください。
関係性のentity1とentity2は、必ずentitiesリストに抽出されたエンティティ名から選んでください。

では、以下のテキストから抽出してください：

{text}
"""

    print(f"  Sending text chunk to Ollama for ERE (model: {model_name})...")
    try:
        # Ollamaにチャットリクエストを送信
        response = ollama_client.chat(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a highly skilled information extraction AI. Extract entities and relations in the specified JSON format only. **Do not include any extra text or markdown formatting like ```json.** Output only the JSON object.",
                },  # システムプロンプトを強化
                {"role": "user", "content": prompt},
            ],
            # 応答がJSONになりやすいようにパラメータを調整 (モデルによるサポート状況に依存)
            options={
                "temperature": 0.0,  # 安定した出力のため低く保つ
                # 'top_p': 0.9,
                # 'num_predict': -1, # 応答全体を出力させる
                # 'stop': ['```'], # これをストップシーケンスに指定すると、```json の始まりで生成を停止させられる場合がある
            },
            # response_format='json' # このパラメータが使えるモデルなら指定（ollamaライブラリやモデルによる）
        )
        raw_json_string = response["message"]["content"]
        print(
            f"  Raw Ollama response received (first 500 chars): {raw_json_string[:500]}..."
        )  # 生の応答の最初をログ出力

        # --- JSONパース処理をより頑丈に ---
        extracted_data = None
        json_string_to_parse = raw_json_string.strip()  # 前後の空白を削除

        # 応答文字列の中から、JSONオブジェクトを示す最初の '{' から最後の '}' までを抽出する
        # これにより、前後の余計なテキストや、```json ... ``` のようなラッパーをある程度無視できる
        first_brace_index = json_string_to_parse.find("{")
        last_brace_index = json_string_to_parse.rfind("}")

        if (
            first_brace_index != -1
            and last_brace_index != -1
            and last_brace_index > first_brace_index
        ):
            # 見つかった '{' から '}' までを抽出
            json_string_to_parse = json_string_to_parse[
                first_brace_index : last_brace_index + 1
            ]
            print(
                f"  Attempting to parse extracted string (first 500 chars): {json_string_to_parse[:500]}..."
            )  # 抽出した文字列をログ出力
        else:
            # '{' と '}' が見つからない、または順序がおかしい場合は、応答全体を試す（たぶん失敗する）
            print(
                "  Warning: Could not find a valid JSON object delimited by {}. Attempting to parse raw response."
            )
            # json_string_to_parse は raw_json_string.strip() のまま

        try:
            # パースを試みる
            extracted_data = json.loads(json_string_to_parse)
            print("  JSON parsed successfully.")

        except json.JSONDecodeError as e:
            print(f"  Error decoding JSON from Ollama response: {e}")
            # パース失敗した文字列とエラー箇所を表示
            print(
                f"  Failed to parse string (first 500 chars): {json_string_to_parse[:500]}..."
            )
            print(f"  Error details: {e}")
            return None  # パース失敗

        # --- 期待するJSON構造（entitiesとrelationsキーが存在するか）を確認 ---
        # パースはできても、内容が期待する形式でない場合がある
        if (
            isinstance(extracted_data, dict)
            and "entities" in extracted_data
            and "relations" in extracted_data
        ):
            # オプション：抽出されたエンティティや関係性の数をログ出力
            # print(f"  Extracted {len(extracted_data.get('entities', []))} entities and {len(extracted_data.get('relations', []))} relations from Ollama.")
            return extracted_data  # 形式が正しい場合は返す
        else:
            # パースはできたが、ルートが辞書でない、またはentities/relationsキーがない場合
            print(
                "Warning: Parsed JSON did not contain expected 'entities' or 'relations' keys at the root, or was not a dictionary."
            )
            # パースできたものの不正だったデータ構造を表示
            print(
                f"  Parsed data structure (first 500 chars): {str(extracted_data)[:500]}..."
            )
            return None  # 形式が不正な場合はNoneを返す

    except ollama.ResponseError as e:
        print(f"  Error during Ollama API call for ERE: {e}")
        return None
    except Exception as e:
        print(f"  An unexpected error occurred during ERE extraction: {e}")
        # 念のため、raw_json_string が取得できていれば表示
        if "raw_json_string" in locals():
            print(
                f"  Partial raw response (first 500 chars): {raw_json_string[:500]}..."
            )
        return None


def add_data_to_neo4j(
    extracted_data: Dict[str, List[Dict[str, str]]],
    driver: GraphDatabase.driver,
    source_info: Dict[str, Any],
):
    """
    抽出したエンティティと関係性をNeo4jに登録する。
    エンティティはMERGEで作成し、関係性もMERGEで作成する（重複防止）。
    MERGE, ON CREATE SET, ON MATCH SET の構文と、関係性MERGE時のノード指定方法を修正。
    """
    if not extracted_data or not driver:
        print("  No data or driver provided for Neo4j addition.")  # ログ追加
        return

    entities = extracted_data.get("entities", [])
    relations = extracted_data.get("relations", [])

    print(
        f"  Attempting to add {len(entities)} entities and {len(relations)} relations to Neo4j..."
    )

    # Neo4jへの書き込みトランザクション関数
    # execute_write に渡す関数はセッションとトランザクションオブジェクト(tx)を受け取る
    def create_graph_tx(tx, entities_to_add, relations_to_add, src_info):
        # --- エンティティ（ノード）の作成またはマージ ---
        # name プロパティを持つ :Entity ノードを一意に扱います。
        # 作成時のみ設定するプロパティと、常に更新する（または作成時のみ設定する）プロパティを分けます。
        entity_merge_query = """
            MERGE (e:Entity {name: $name})
            ON CREATE SET
                e.type = $type,             // 作成時のみタイプを設定
                e.source_file = $source_file, // 作成時のみソースを設定
                e.creation_time = timestamp() // 作成時のみ作成時間を設定
            ON MATCH SET
                 e.source_file = $source_file // 既存ノードの場合もソースファイル情報は更新しても良い（複数ソースに登場する場合など）
            RETURN e // マージ/作成されたノードを返す（オプション、デバッグ用にも）
        """
        for entity in entities_to_add:
            entity_name = entity.get("name")
            entity_type = entity.get("type", "Unknown")

            if not entity_name:
                print(f"Warning: Skipping entity with no name during MERGE: {entity}")
                continue

            try:
                # エンティティのマージ実行
                tx.run(
                    entity_merge_query,
                    name=entity_name,
                    type=entity_type,
                    source_file=src_info.get("source_file"),
                )
                # print(f"  Merged entity: {entity_name}") # 成功ログ（詳細すぎる場合はオフに）

            except Exception as e:
                # 個別のエンティティ登録エラーを捕捉
                print(f"Error merging entity '{entity_name}' into Neo4j: {e}")
                # エラーが発生してもトランザクションは継続される（設定による）か、全体がロールバックする

        # --- 関係性（エッジ）の作成またはマージ ---
        # 関係性の両端ノードを MATCH で見つけてから、関係性を MERGE します。
        # これにより、ノードの存在が保証されます。
        relationship_merge_query_template = """ // f-string テンプレート
            MATCH (e1:Entity {name: $entity1_name}) // 関係性の開始ノードを見つける
            MATCH (e2:Entity {name: $entity2_name}) // 関係性の終了ノードを見つける
            MERGE (e1)-[r:`{relation_type_cleaned}`]->(e2) // 両端ノード間に指定タイプの関係性をマージ
            ON CREATE SET
                r.source_file = $source_file, // 関係性作成時のみソースを設定
                r.creation_time = timestamp() // 関係性作成時のみ作成時間を設定
            RETURN r // マージ/作成された関係性を返す（オプション）
        """
        for relation in relations_to_add:
            entity1_name = relation.get("entity1")
            relation_type = relation.get("relation")
            entity2_name = relation.get("entity2")

            if not entity1_name or not relation_type or not entity2_name:
                print(f"Warning: Skipping incomplete relation during MERGE: {relation}")
                continue

            # 関係性タイプをCypherで使える形式に整形（特殊文字を安全な文字に置換）
            # バッククォート ` ` で囲めば一部特殊文字も使えますが、安全のため置換を推奨
            cleaned_relation_type = (
                relation_type.strip()
                .replace(" ", "_")
                .replace("-", "_")
                .replace(":", "_")
                .replace(">", "_")
                .replace("<", "_")
                .replace(".", "_")
                .replace("(", "_")
                .replace(")", "_")
                .replace("[", "_")
                .replace("]", "_")
                .replace("{", "_")
                .replace("}", "_")
                .replace('"', "_")
                .replace("'", "_")
                .replace("\\", "_")
                .replace("/", "_")
                .replace("?", "_")
                .replace("!", "_")
                .replace("@", "_")
                .replace("#", "_")
                .replace("$", "_")
                .replace("%", "_")
                .replace("^", "_")
                .replace("&", "_")
                .replace("*", "_")
                .replace("+", "_")
                .replace("=", "_")
                .replace("|", "_")
                .replace(";", "_")
                .replace(",", "_")
                .upper()
            )

            # 整形後の関係性タイプが空文字列にならない、またはNeo4jの制約に違反しないかチェック（例：タイプ名は空にできない）
            if not cleaned_relation_type:
                print(
                    f"Warning: Skipping relation with invalid or empty type after cleaning: {relation_type}"
                )
                continue
            # 関係性タイプが全て数字になるのも避ける（Cypherの制約）
            if cleaned_relation_type.isdigit():
                cleaned_relation_type = "REL_" + cleaned_relation_type
            # タイプ名の先頭が数字になるのも避ける
            if cleaned_relation_type and cleaned_relation_type[0].isdigit():
                cleaned_relation_type = "_" + cleaned_relation_type

            try:
                # 関係性のマージ実行 (Cypherテンプレートに整形済みタイプ名を埋め込む)
                relationship_merge_query = relationship_merge_query_template.format(
                    relation_type_cleaned=cleaned_relation_type
                )
                tx.run(
                    relationship_merge_query,
                    entity1_name=entity1_name,
                    entity2_name=entity2_name,
                    source_file=src_info.get("source_file"),
                )
                # print(f"  Merged relation: {entity1_name}-[{cleaned_relation_type}]->{entity2_name}") # 成功ログ

            except Exception as e:
                # 個別の関係性登録エラーを捕捉
                print(
                    f"Error merging relation '{entity1_name}'-['{relation_type}']->'{entity2_name}' into Neo4j: {e}"
                )
                # トランザクションが失敗した場合、ここで個別のエラーメッセージを出すのは難しいかもしれない

    # --- トランザクションの実行 ---
    try:
        # execute_write はトランザクション内で指定した関数を実行し、成功すればコミット、失敗すればロールバックする
        driver.session().execute_write(
            create_graph_tx, entities, relations, source_info
        )
        # execute_write が例外を投げなければ成功
        print(
            f"  Transaction for adding {len(entities)} entities and {len(relations)} relations completed successfully."
        )

    except Exception as e:
        # execute_write 自体が投げた例外 (トランザクション全体が失敗した場合など)
        print(
            f"Error executing Neo4j transaction for {src_info.get('source_file', 'Unknown Source')}: {e}"
        )


# --- メイン処理 ---


def preprocess_for_graph(
    data_dir: str = DATA_DIR,
    ere_model: str = ERE_MODEL,
    neo4j_uri: str = URI,
    neo4j_user: str = USER,
    neo4j_password: str = PASSWORD,
):
    """
    指定ディレクトリのファイルを読み込み、GraphRAG向けに処理してNeo4jに格納する。
    """
    print(f"--- Starting Graph Preprocessing for directory: {data_dir} ---")

    driver = None
    ollama_client = None

    try:
        # Neo4jドライバーを作成
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        # 接続確認
        driver.verify_connectivity()
        print("Neo4j connection successful.")

        # Ollamaクライアント初期化
        ollama_client = ollama.Client()
        print(f"Using Ollama model for ERE: {ere_model}")

    except Exception as e:
        print(f"Error connecting to services: {e}")
        print("Cannot proceed with graph preprocessing.")
        # driverやclientがNoneの場合でも後続処理に進むため、ここでreturn
        if driver:
            driver.close()
        return  # エラーが発生した場合は処理を中断

    # データディレクトリが存在しない場合はエラー
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found at {data_dir}")
        driver.close()
        return

    # ディレクトリ内のファイルを走査
    processed_files_count = 0
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)

        # ディレクトリではなくファイルであることを確認
        if os.path.isfile(filepath):
            print(f"Processing file for graph: {filename}")

            # ファイルコンテンツ読み込み
            content = read_file_content(filepath)
            if not content:
                print(f"Skipping empty or unreadable file: {filename}")
                continue

            # ファイル全体を一つの大きなチャンクとして扱うか、小さく分割するか
            # EREの精度はチャンクサイズに影響される。ここではシンプルにファイル全体を渡す（長いとLLMのコンテキストサイズ制限に注意）
            # または、preprocess_data_rag.py のチャンク分割を再利用しても良い
            # ここでは簡単のためファイル全体をEREにかける例
            chunks_to_process = [content]  # ファイル全体を一つのチャンクとして扱う
            # または preprocess_data_rag.py から simple_chunk_text をインポートして使う場合：
            # chunks_to_process = simple_chunk_text_for_ere(content) # より小さなチャンクで処理

            extracted_count = 0
            for i, chunk in enumerate(chunks_to_process):
                print(
                    f"  Extracting entities/relations from chunk {i + 1}/{len(chunks_to_process)} (via Ollama)..."
                )
                extracted_data = extract_entities_relations_with_ollama(
                    chunk, ollama_client, ere_model
                )

                if extracted_data:
                    # Neo4jに登録
                    source_info = {
                        "source_file": filename,
                        "chunk_id": i,  # ファイル全体の場合は 0
                        # 必要なら他の情報も追加
                    }
                    add_data_to_neo4j(extracted_data, driver, source_info)
                    extracted_count += len(extracted_data.get("entities", [])) + len(
                        extracted_data.get("relations", [])
                    )  # 大まかな登録要素数

            if extracted_count > 0:
                print(
                    f"  Successfully extracted and prepared {extracted_count} entities/relations for Neo4j."
                )
            else:
                print(f"  No entities or relations extracted from {filename}.")

            processed_files_count += 1

    print(
        f"--- Graph Preprocessing Complete. Processed {processed_files_count} file(s). ---"
    )
    print("Please check your Neo4j Browser to visualize the graph data.")

    # 処理完了後にドライバーを閉じる
    if driver:
        driver.close()


# --- スクリプトとして実行された場合の処理 ---
if __name__ == "__main__":
    # 事前入力データディレクトリが存在しない場合は警告
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found.")
        print(
            f"Please create the '{DATA_DIR}' directory and place your source files inside."
        )
    # ディレクトリ内のファイルが空でないか確認
    elif not any(
        os.path.isfile(os.path.join(DATA_DIR, name)) for name in os.listdir(DATA_DIR)
    ):
        print(f"Data directory '{DATA_DIR}' is empty.")
        print(
            f"Please place your .txt or .json files inside the '{DATA_DIR}' directory."
        )
    else:
        # Graph前処理を実行
        preprocess_for_graph(DATA_DIR, ERE_MODEL, URI, USER, PASSWORD)
        print("\nGraph preprocessing script finished.")
