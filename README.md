## はじめに

OpenAI が公開した gpt-oss を Apple Silicon 搭載の Mac では GPU（Metal）を Enable にした llama.cpp で動作させる


## 1. llama.cpp のビルドとセットアップ　（Metal 対応)

まずは、GPU に対応した llama.cpp を取得し、Metal オプションを有効にしてビルドします。

### ソースコードの取得

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
```

### ビルド（Metal を有効化）

Metal オプションを有効にすることで、Apple Silicon の GPU を活用した LLM 実行が可能になります。

```bash
mkdir -p build
cd build
cmake .. -DLLAMA_METAL=ON
cmake --build . --config Release
```

## 2. モデルの準備と配置

llama.cpp で利用する GGUF フォーマットのモデルをダウンロード・配置します。

### モデルのダウンロード

gpt-oss のモデルは HuggingFace で公開されています。
Hugging Face 上のモデルをコマンドラインまたはブラウザから取得します。

- [オリジナルのモデル gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b)
- [オリジナルのモデル gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b)
- [GUFF形式のモデル gpt-oss-20b](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF)

#### コマンドラインの場合

```bash
cd ../models
# gpt-oss-20b-mxfp4.gguf を取得する
wget https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-mxfp4.gguf
cd ../build
```

## 3. OpenAI 互換サーバの起動

モデルを配置したら、llama-server を起動し、OpenAI API と互換性のあるエンドポイントを立てます

```bash
./bin/llama-server \
  -m ../models/gpt-oss-20b-mxfp4.gguf \
  --port 8080
```

| オプション   | 説明    | 詳細・効果   |
|---|----|-----|
| `./bin/llama-server`| llama.cpp に含まれる HTTP サーバープログラム | モデルを読み込み、OpenAI API 互換のエンドポイント (`/v1/chat/completions` など) を提供する|
| `-m ../models/gpt-oss-20b-mxfp4.gguf` | 使用するモデルファイルのパス  | GPT-OSS-20B モデル（mxfp4量子化、GGUF形式）。相対パスで `../models` ディレクトリ内のファイルを指定している|
| `--port 8080`                  | サーバーの待ち受けポート番号を指定  | この場合、`http://localhost:8080` でAPIへアクセス可能|

起動時ログで処理がGPUにオフロードされている(METALが利用されている) ことが確認できる

```bash
load_tensors: offloading 24 repeating layers to GPU
load_tensors: offloading output layer to GPU
load_tensors: offloaded 25/25 layers to GPU
ggml_metal_init: GPU name:   Apple M3 Pro
ggml_metal_init: GPU family: MTLGPUFamilyApple9  (1009)
ggml_metal_init: GPU family: MTLGPUFamilyCommon3 (3003)
ggml_metal_init: GPU family: MTLGPUFamilyMetal3  (5001)
```

## 4. Curl によるテスト実行とレスポンス確認

サーバが正常に起動したら、以下の curl コマンドで推論を実行できます

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-20b",
    "messages": [
      {"role": "user", "content": "PythonでHello Worldを表示するコードを書いてください。"}
    ],
    "temperature": 0.7,
    "max_tokens": 256,
    "stream": false
  }' | jq -r '.choices[0].message.content'
```

このコマンドにより、次のような応答が得られます。

````bash
<|channel|>analysis<|message|>We need to provide a Python code snippet that prints "Hello World". The user is Japanese: "PythonでHello Worldを表示するコードを書いてください。" They want a code. Provide code and maybe explanation. Keep it concise.<|start|>assistant<|channel|>final<|message|>```python
# Python で Hello World を表示するシンプルなコード

print("Hello World")
```
````


## 5. 簡易チャットアプリ

```bash
git clone https://github.com/aRaikoFunakami/langchain_gpt_oss.git
uv sync
uv run python main.py
```