# %%
import pathlib

import duckdb
import openai
import pandas as pd
import json
from sklearn.metrics import classification_report, f1_score, balanced_accuracy_score
import tiktoken

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# %%
con = duckdb.connect(":memory:")

# %%
con.execute("SELECT COUNT(*) FROM '../data/wildchat.parquet'").fetchone()

# %%
con.execute("SELECT COUNT(DISTINCT conversation_hash) FROM '../data/wildchat.parquet'").fetchone()

# %%
con.execute("SELECT COUNT(*) FROM '../data/wildchat.parquet' WHERE country = 'United States' AND role = 'user' AND language = 'English'").fetchone()

# %%
con.execute("SELECT COUNT(DISTINCT conversation_hash) FROM '../data/wildchat.parquet' WHERE country = 'United States' AND role = 'user' AND language = 'English'").fetchdf()

# %%
con.execute("SELECT classification, COUNT(*) FROM '../data/sample_for_annotation_annotated.csv' GROUP BY classification").fetchdf()

# %%
with open("../data/searched_news.txt", "r") as f:
    records = f.read()

print(len(records.split("---")) - 1)

# %%
llm = openai.OpenAI()

# %%
with open("./prompts/classification.txt", "r") as f:
    prompt = f.read()

# %%
annotations = con.execute("SELECT content, classification FROM '../data/sample_for_annotation_annotated.csv'").fetchdf()

# %%
searched = pd.DataFrame([r.strip() for r in records.split("---")[:-1]], columns=["content"])
searched["classification"] = 1

# %%
annotations = pd.concat([annotations, searched])

# %%
annotations.classification.value_counts()

# %%
outputs_mini = []
outputs_o = []
for _, s in tqdm(annotations.iterrows(), total=len(annotations)):
    resp = llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"[MESSAGE]{s.content}[\\MESSAGE]"},
        ]
    )
    outputs_mini.append(resp.choices[0].message.content)
    resp = llm.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"[MESSAGE]{s.content}[\\MESSAGE]"},
        ]
    )
    outputs_o.append(resp.choices[0].message.content)

# %%
annotations["gpt-4o-mini"] = outputs_mini
annotations["gpt-4o"] = outputs_o
annotations.to_csv("../data/sample_for_annotation_annotated_llm.csv", index=False)

# %%
def extract_classification(output):
    clf = output.split("\n")[0]
    try:
        return int(clf)
    except ValueError:
        return None

# %%
annotations["clf_gpt-4o-mini"] = annotations["gpt-4o-mini"].apply(extract_classification)
annotations["clf_gpt-4o"] = annotations["gpt-4o"].apply(extract_classification)

# %%
# set to 0 if not found (1 record)
annotations.loc[annotations["clf_gpt-4o"].isna(), "clf_gpt-4o"] = 0

# %%
annotations["clf_gpt-4o-mini"].value_counts()

# %%
annotations["clf_gpt-4o"].value_counts()

# %%
print(classification_report(annotations["classification"], annotations["clf_gpt-4o-mini"]))

# %%
print(classification_report(annotations["classification"], annotations["clf_gpt-4o"]))

# %%
print(f1_score(annotations["classification"], annotations["clf_gpt-4o-mini"], average="weighted"))
print(f1_score(annotations["classification"], annotations["clf_gpt-4o"], average="weighted"))
print(balanced_accuracy_score(annotations["classification"], annotations["clf_gpt-4o-mini"]))
print(balanced_accuracy_score(annotations["classification"], annotations["clf_gpt-4o"]))

# %%
enc = tiktoken.encoding_for_model("gpt-4o-mini")

# %%
# %%
all_user_messages = con.execute("SELECT content FROM '../data/wildchat.parquet' WHERE country = 'United States' AND role = 'user' AND language = 'English'").fetchdf()
all_tokens = sum([len(enc.encode(m, disallowed_special=())) for m in all_user_messages.content])
print(f"Total tokens: {all_tokens}")
print(f"Cost: ${all_tokens * 0.15 / 1_000_000:.2f}")

# %%
# Output
all_outputs = con.execute("""SELECT "gpt-4o-mini" FROM '../data/sample_for_annotation_annotated_llm.csv'""").fetchdf()
all_tokens = sum([len(enc.encode(m, disallowed_special=())) for m in all_outputs["gpt-4o-mini"]])
print(f"Total tokens: {all_tokens}")
cost_per_record = all_tokens * 0.6 / 1_000_000 / len(all_outputs)
print(f"Cost per record: ${cost_per_record:.2f}")
print(f"Total cost: ${cost_per_record * len(all_user_messages):.2f}")

# %%
full_sample = con.execute("""SELECT conversation_hash, content FROM '../data/wildchat.parquet'
            WHERE country = 'United States' AND role = 'user' AND language = 'English'
            """).fetch_df()

# %%
# Calculate the number of chunks needed
num_chunks = len(full_sample) // 40000 + (1 if len(full_sample) % 40000 != 0 else 0)

# Create a list to store the chunks
chunks = []

# Split the dataframe into chunks
for i in range(num_chunks):
    start_idx = i * 40000
    end_idx = min((i + 1) * 40000, len(full_sample))
    chunk = full_sample.iloc[start_idx:end_idx].copy()
    chunks.append(chunk)

print(f"Number of chunks created: {len(chunks)}")
print(f"Rows in first chunk: {len(chunks[0])}")
print(f"Rows in last chunk: {len(chunks[-1])}")

# %%
def format_for_batch_submission(row):
    obj = {
        "custom_id": str(row.name),
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"[MESSAGE]{row.content}[\\MESSAGE]"},
            ]
        }
    }

    return obj

# %%
# apply formatting to each chunk
chunks = [chunk.apply(format_for_batch_submission, axis=1) for chunk in chunks]

# %%
# save each chunk as a .jsonl file
for i, chunk in enumerate(chunks):
    with open(f"../data/batches/batch_{i}.jsonl", "w") as f:
        for _, s in chunk.items():
            f.write(json.dumps(s) + "\n")

# %%
# list all .jsonl files in the batches folder
batch_files = list(pathlib.Path("../data/batches").glob("*.jsonl"))
print(f"Number of batch files: {len(batch_files)}")

batch_input_refs = []

for batch_file in batch_files:
    batch_input_file = llm.files.create(
        file=open(batch_file, "rb"),
        purpose="batch"
    )
    batch_input_refs.append(batch_input_file.id)

# %%
for b in batch_input_refs[1:]:
    llm.batches.create(
        input_file_id=b,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "Batch annotation of user messages in WildChat"}
    )

# %%
batches = [i for i in llm.batches.list().data if i.status == "completed"]

# %%
it = 0

for b in batches:
    input_file = llm.files.content(b.input_file_id).read()
    output_file = llm.files.content(b.output_file_id).read()

    with open(f"../data/batches_from_openai/{it}_input.jsonl", "wb") as f:
        f.write(input_file)

    with open(f"../data/batches_from_openai/{it}_output.jsonl", "wb") as f:
        f.write(output_file)

    it += 1
# %%
files = []
for i in range(0,7):
    input_file = pd.read_json(f"../data/batches_from_openai/{i}_input.jsonl", lines=True)
    output_file = pd.read_json(f"../data/batches_from_openai/{i}_output.jsonl", lines=True)

    input_file["message_body"] = input_file.body.apply(lambda x: x["messages"][1]["content"])
    input_file = input_file[["custom_id", "message_body"]]

    output_file = output_file[output_file.error.isna()]
    output_file["response_body"] = output_file.response.apply(lambda x: x["body"]["choices"][0]["message"]["content"])
    output_file = output_file[["custom_id", "response_body"]]

    all_data = input_file.merge(output_file, on="custom_id")

    files.append(all_data)

all_data = pd.concat(files)
all_data["classification"] = all_data.response_body.apply(extract_classification)
print(all_data[all_data.classification.isna()])
print(len(all_data[all_data.classification.isna()]))
all_data.loc[all_data["classification"].isna(), "classification"] = 0
print(len(all_data))
all_data.to_parquet("../data/batches_from_openai/all_data.parquet")
