from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import ast
import torch
import numpy as np
import streamlit as st
import io
import re
from transformers import GPT2LMHeadModel, GPT2Tokenizer


@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained("CSC671/gpt1-ioc-model_meta_cve")
    model = GPT2LMHeadModel.from_pretrained("CSC671/gpt1-ioc-model_meta_cve")
    return tokenizer, model
    
login(st.secrets["huggingface"]["token"])
# === Classification parameters ===
HYPOTHESIS_TEMPLATE = "This example is {}."  # template for zero-shot hypothesis

def format_ioc_for_prediction(row):
    tag_str = ", ".join(row["tags"])
    meta_items = [f"{k}: {v}" for k, v in row["metadata"].items()]
    meta_str = " | ".join(meta_items)
    cves = ", ".join(row["cve"]) if row["cve"] else "None"
    vpn = row["vpn"]
    spoof = row["spoofable"]
    classi = row["classification"]
    return (
        f"<start>\n"
        f"Tags: {tag_str}\n"
        f"Metadata: {meta_str}\n"
        f"CVE: {cves}\n"
        f"VPN: {vpn}\n"
        f"Spoofable: {spoof}\n"
        f"Classification: {classi}\n"
        f"TTPs:"
    )


def reconstruct_and_extract_ttps(generated_text):
    cleaned = generated_text.replace(",", " ").replace("'", "").replace("\n", " ")
    cleaned = re.sub(r"\s+", "", cleaned)
    matches = re.findall(r"T\d{4}\.\d{3}", cleaned)
    return list(dict.fromkeys(matches))
def safe_parse(val, fallback=None):
    try:
        # Check if value is string and starts like a list or dict
        if isinstance(val, str) and (val.startswith("{") or val.startswith("[")):
            return ast.literal_eval(val)
    except Exception:
        pass
    return fallback if fallback is not None else {}


def parse_gn_csv_to_ioc_data(uploaded_file):
    df = pd.read_csv(uploaded_file)

    ioc_data_list = []

    for _, row in df.iterrows():
        try:
            # Check if the CSV has the old structure or the new simplified structure
            if 'metadata' in df.columns and 'raw_data' in df.columns and 'cve' in df.columns:
                # Parse metadata and raw_data fields (old structure)
                meta_dict = safe_parse(row['metadata'], {})
                raw_dict = safe_parse(row['raw_data'], {})

                ioc_data = {
                    "ip": row['ip'],
                    "metadata": {
                        **meta_dict,
                        "scan": raw_dict.get("scan", []),
                        "sensor_hits": meta_dict.get("sensor_hits", None),
                    },
                    "tags": safe_parse(row['tags'], []),
                    "cves": safe_parse(row['cve'], []),
                }
                print(ioc_data)
            else:
                # New simplified structure with just ip and tags
                ioc_data = {
                    "ip": row['ip'],
                    "metadata": {},
                    "tags": safe_parse(row['tags'], []),
                    "cves": [],
                }
            ioc_data_list.append(ioc_data)
        except Exception as e:
            print(f"Error parsing row: {e}")
            continue

    return ioc_data_list


def build_ttp_prompt(ioc_data):
    ip = ioc_data.get("ip", "Unknown")
    metadata = ioc_data.get("metadata", {})
    tags = ioc_data.get("tags", [])
    cves = ioc_data.get("cves", [])

    # Base prompt
    prompt = f"""
You are a cybersecurity threat analyst. Given the following IOC and its context, assign the most likely MITRE ATT&CK techniques (TTPs). Be specific.

IOC: {ip}
Metadata:
- ASN: {metadata.get('asn', 'Unknown')}
- City: {metadata.get('city', 'Unknown')}, {metadata.get('country', 'Unknown')}
- Org: {metadata.get('org', 'Unknown')}
- Sensor Hits: {metadata.get('sensor_hits', 'Unknown')}
- Scan Behavior: {", ".join(f"TCP port {s['port']}" for s in metadata.get('scan', []))}
- Tags: {", ".join(tags) if tags else 'None'}
- CVEs: {", ".join(cves) if cves else 'None'}
"""

    # If tags exist, specifically ask for TTPs based on those tags
    if tags:
        prompt += f"""
Based on the imported tags ({', '.join(tags)}), identify the most relevant MITRE ATT&CK techniques (TTPs) that match these behaviors or indicators.
"""

    prompt += "\nTTP:\n"
    return prompt.strip()


# Load MITRE ATT&CK techniques and tactics data
techniques_df = pd.read_csv("mitre_attack_techniques.csv")
tactics_df = pd.read_csv("mitre_attack_tactics.csv")
# Prepare candidate labels for zero-shot classification
technique_labels = techniques_df["Technique Name"].tolist()
# Prepare tactic labels for zero-shot classification
tactic_labels = tactics_df["Tactic Name"].tolist()

# Load SecBERT model and tokenizer from jackaduma/SecBERT
model_name = "jackaduma/SecBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Create a custom classifier function for SecBERT
def secbert_classifier(text, candidate_labels, multi_label=False, top_k=1):
    # Prepare inputs for each candidate label
    inputs = []
    for label in candidate_labels:
        # Format text with hypothesis template
        formatted_text = text + " " + HYPOTHESIS_TEMPLATE.format(label)
        encoded = tokenizer(formatted_text, truncation=True, max_length=512, padding=True, return_tensors="pt")
        inputs.append(encoded)

    # Get predictions for each label
    scores = []
    with torch.no_grad():
        for encoded in inputs:
            # Move inputs to the same device as the model
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)
            # Get the probability for the positive class (index 1)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            score = probabilities[0, 1].item()  # Probability of positive class
            scores.append(score)

    # Convert to numpy array for easier manipulation
    scores = np.array(scores)

    # Get top-k indices
    if multi_label:
        # For multi-label, return all labels above threshold
        top_indices = np.argsort(scores)[-top_k:][::-1]
    else:
        # For single-label, return only the top label
        top_indices = np.argsort(scores)[-1:][::-1]

    # Prepare results in the same format as the pipeline
    labels = [candidate_labels[i] for i in top_indices]
    scores_list = [scores[i] for i in top_indices]

    return {"labels": labels, "scores": scores_list}



st.title('IoC to TTPs using LLMs')

# Upload CSV file
THRESHOLD = st.slider("Confidence Threshold", 0.0, 1.0, 0.6, step=0.05)
TOP_K = st.number_input("Top-K Predictions", min_value=1, max_value=50, value=10)
TEMPERATURE = st.slider("Temperature (for GPT-2)", 0.0, 2.0, 0.7, step=0.05)
MAX_NEW_TOKENS = st.number_input("Max New Tokens (for GPT-2)", min_value=1, max_value=1000, value=300)

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:

    uploaded_bytes = uploaded_file.read()
    
    # Create two separate file-like objects from the same bytes
    file_copy1 = io.BytesIO(uploaded_bytes)
    file_copy2 = io.BytesIO(uploaded_bytes)
    st.subheader("Predicting using the SecBert Model....")
    ioc_data_list = parse_gn_csv_to_ioc_data(file_copy1)
    print(len(ioc_data_list))
    
    for ioc_data in ioc_data_list:
        prompt = build_ttp_prompt(ioc_data)
        # Classify techniques using SecBERT
        tech_res = secbert_classifier(
            prompt,
            candidate_labels=technique_labels,
            multi_label=True,
            top_k=TOP_K
        )
        # Ensure we always get at least one prediction by taking the top one if none exceed threshold
        pred_tech = [label for label, score in zip(tech_res["labels"], tech_res["scores"]) if score > THRESHOLD]
        if not pred_tech and tech_res["labels"]:  # If no predictions exceed threshold but we have labels
            pred_tech = [tech_res["labels"][0]]  # Take the top prediction
        # Map technique names to IDs and handle sub-techniques
        pred_tech_ids = []
        for t in pred_tech:
            st.write("Loading the IoC")
            # Find the technique ID for this technique name
            tech_id_series = techniques_df.loc[techniques_df["Technique Name"] == t, "Technique ID"]
            if not tech_id_series.empty:
                tech_id = tech_id_series.iloc[0]
                # Check if this is a sub-technique (ID starts with a comma)
                if isinstance(tech_id, str) and tech_id.startswith(','):
                    # Find the parent technique (the last main technique ID before this sub-technique)
                    parent_idx = techniques_df.index[techniques_df["Technique Name"] == t].tolist()[0]
                    # Go backwards until we find a main technique (ID starts with 'T')
                    for idx in range(parent_idx-1, -1, -1):
                        potential_parent_id = techniques_df.iloc[idx]["Technique ID"]
                        if isinstance(potential_parent_id, str) and potential_parent_id.startswith('T'):
                            parent_id = potential_parent_id
                            parent_name = techniques_df.iloc[idx]["Technique Name"]
                            # Format as (parent_id, parent_name, sub_id, sub_name)
                            pred_tech_ids.append((parent_id, parent_name, tech_id, t))
                            break
                else:
                    # This is a main technique, just add the ID
                    pred_tech_ids.append(tech_id)
            else:
                # If we can't find the technique, add a placeholder
                pred_tech_ids.append("Unknown")
        # Classify tactics using SecBERT
        tac_res = secbert_classifier(
            prompt,
            candidate_labels=tactic_labels,
            multi_label=True,
            top_k=TOP_K
        )
        # Ensure we always get at least one prediction by taking the top one if none exceed threshold
        pred_tac = [label for label, score in zip(tac_res["labels"], tac_res["scores"]) if score > THRESHOLD]
        if not pred_tac and tac_res["labels"]:  # If no predictions exceed threshold but we have labels
            pred_tac = [tac_res["labels"][0]]  # Take the top prediction
        # Map tactic names to IDs
        pred_tac_ids = [
            tactics_df.loc[tactics_df["Tactic Name"] == t, "Tactic ID"].iloc[0]
            for t in pred_tac
        ]
        st.write(f"IOC: {ioc_data['ip']}")
        st.write(ioc_data['tags'][:5])
        st.write("Predicted Techniques")
        for i, tech_id in enumerate(pred_tech_ids):
            if isinstance(tech_id, tuple):
                parent_id, parent_name, sub_id, sub_name = tech_id
                st.markdown(f"- **{sub_id}** ({sub_name}) — Sub-technique of **{parent_id}** ({parent_name})")
            else:
                st.markdown(f"- **{tech_id}** ({pred_tech[i]})")

        # Format tactics for display
        st.write("Predicted Tactics")
        for ttp_id, tactic in zip(pred_tac_ids, pred_tac):
            st.markdown(f"- **{ttp_id}** ({tactic})")
        
    st.subheader("Predicting using Fine-Tuned GPT2")
    

    tokenizer, model = load_model()
    df = pd.read_csv(file_copy2)
    df['tags'] = df['tags'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    df['cve'] = df['cve'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    df['metadata'] = df['metadata'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df = df[['ip','tags', 'metadata', 'cve', 'vpn', 'spoofable', 'classification']]
    for i in range(len(df)):
        input_text = format_ioc_for_prediction(df.iloc[i])
        inputs = tokenizer(input_text, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            top_p=0.95,
            temperature=TEMPERATURE,
            pad_token_id=tokenizer.eos_token_id
        )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        ttp_block = decoded.split("TTPs:")[-1].split("<end>")[0].strip()
        ttp_predictions = reconstruct_and_extract_ttps(ttp_block)
        st.write(f"IOC {df['ip'].iloc[i]} Predicted TTPs:")
        for ttp in ttp_predictions:
            st.markdown(f"- **{ttp}**")
