# llm_generate.py

from typing import Optional

import ollama

# --- LLM回答生成 ---


def generate_response(
    query: str, context: str, model_name: str, ollama_client: ollama.Client
) -> Optional[str]:
    print(f"モデル： '{model_name}'")

    # システムプロンプトとユーザープロンプトを組み合わせる
    system_prompt = (
        "あなたはユーザーの質問に対し、提供された【参照情報】または【Graph参照情報】のみを用いて回答するAIアシスタントです。\n"
        "【参照情報】または【Graph参照情報】に含まれていない内容については推測で答えてください。\n"
    )

    user_prompt = f"質問：{query}\n\n{context}"

    try:
        response = ollama_client.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        llm_response = response["message"]["content"]
        print("生成成功")
        return llm_response

    except ollama.ResponseError as e:
        print(f"Error generating response with Ollama ({model_name}): {e}")
        return f"回答生成中にエラーが発生 (モデル: {model_name}): {e}"
    except Exception as e:
        print(
            f"An unexpected error occurred during response generation ({model_name}): {e}"
        )
        return f"回答生成中にエラーが発生 (モデル: {model_name}): {e}"
