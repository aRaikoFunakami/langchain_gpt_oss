## はじめに

[OpenAI が公開した gpt-oss](https://openai.com/ja-JP/index/introducing-gpt-oss/) を Apple Silicon 搭載の Mac では GPU（Metal）を Enable にした llama.cpp で動作させます。

OpenAI によると「gpt-oss-20b モデルは、一般的なベンチマークで OpenAI o3‑mini と同様の結果を出し、わずか16 GBのメモリを搭載したエッジデバイスで実行できる」とのことです。


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
cd langchain_gpt_oss
uv sync
uv run python main.py
```

コマンドラインが立ち上がります

```bash
Chat started. Type 'exit' to quit.
You: 
```

### サンプル出力1

```bash
You: こんにちは
Assistant: こんにちは！どんなお手伝いが必要ですか？
```

実行時間 (サーバー側のログ) 

```bash
prompt eval time =     285.82 ms /    40 tokens (    7.15 ms per token,   139.95 tokens per second)
       eval time =    2239.42 ms /    94 tokens (   23.82 ms per token,    41.98 tokens per second)
      total time =    2525.24 ms /   134 tokens
```


### サンプル出力2

```bash
You: スターウォーズの監督を教えて
Assistant: スターウォーズシリーズの主要な監督は以下の通りです（公開順・主要作品別にまとめました）。

| 作品 | 公開年 | 監督 |
|------|--------|------|
| **『スター・ウォーズ エピソード4/新たなる希望』** | 1977 | ジョージ・ルーカス |
| **『スター・ウォーズ エピソード5/帝国の逆襲』** | 1980 | リチャード・マーロウ |
| **『スター・ウォーズ エピソード6/ジェダイの帰還』** | 1983 | リチャード・マーロウ |
| **『スター・ウォーズ エピソード1/ファントム・メナス』** | 1999 | ジョージ・ルーカス |
| **『スター・ウォーズ エピソード2/クローンの攻撃』** | 2002 | ジョージ・ルーカス |
| **『スター・ウォーズ エピソード3/シスの復讐』** | 2005 | ジョージ・ルーカス |
| **『スター・ウォーズ エピソード7/フォースの覚醒』** | 2015 | J.J. アブラムズ |
| **『スター・ウォーズ エピソード8/最後のジェダイ』** | 2017 | Rian Johnson |
| **『スター・ウォーズ エピソード9/スカイウォーカーの夜明け』** | 2019 | ジョージ・ルーカス（監督はリード・レイザーが制作に関わるが、公式に監督はジョージ・ルーカス） |
| **『レイ・オブ・ハッピー』**（スピンオフ） | 2018 | レイ・オブ・ハッピー |  
| **『フォースの暗闇』**（スピンオフ） | 2018 | ジョン・ファブレック |

> **注**  
> - 公式に「監督」と認定されるのは上記表の通りです。  
> - ストーリーラインや制作背景に関わる他の重要人物（プロデューサー、脚本家など）は別途多く存在します。  

もし「スターウォーズの監督」を特定の映画（例：オリジナル三部作）に絞って知りたい場合は、その作品名を教えていただければさらに詳しく回答します。
```


実行時間 (サーバー側のログ) 

```bash
prompt eval time =    3099.14 ms /  1360 tokens (    2.28 ms per token,   438.83 tokens per second)
       eval time =   20143.76 ms /   651 tokens (   30.94 ms per token,    32.32 tokens per second)
      total time =   23242.90 ms /  2011 tokens
```