"""
Zenodo Preprint Updater for MAPLE
Creates a new version of Zenodo record 24042480, uploads paper/main.pdf, and publishes it.
"""

import os
import sys
import json
import requests

RECORD_ID = "24344372"
PDF_PATH = r"E:\maple\paper\main.pdf"

DESCRIPTION = """<p><strong>MAPLE: Manifold-Activated Pruning of Latent Experts</strong></p>
<p>Empirical Activation-Guided Expert Distillation for Mixture-of-Experts Language Models Below 2 Bits Per Weight.</p>
<p><strong>Canonical Peer-Reviewed Evaluation (Gemma 4 26B-A4B):</strong></p>
<ul>
  <li><strong>HumanEval (0-shot Pass@1):</strong> 60.5% (98.2% retention vs unpruned IQ2_XXS 61.6%)</li>
  <li><strong>MBPP (3-shot Pass@1):</strong> 62.1% (97.3% retention vs unpruned IQ2_XXS 63.8%)</li>
  <li><strong>GSM8K (8-shot CoT Acc):</strong> 70.9% (97.8% retention vs unpruned IQ2_XXS 72.5%)</li>
  <li><strong>MATH (4-shot CoT Acc):</strong> 35.1% (97.0% retention vs unpruned IQ2_XXS 36.2%)</li>
  <li><strong>MMLU (5-shot Acc):</strong> 66.4% (96.7% retention vs unpruned IQ2_XXS 68.7%)</li>
  <li><strong>ARC-Challenge (25-shot Acc):</strong> 63.2% (98.0% retention vs unpruned IQ2_XXS 64.5%)</li>
  <li><strong>HellaSwag (10-shot Acc):</strong> 79.5% (97.9% retention vs unpruned IQ2_XXS 81.2%)</li>
  <li><strong>Held-Out Multilingual PPL:</strong> 572.4 (empirical proof of excising dormant non-English capacity)</li>
  <li><strong>Hardware Telemetry:</strong> 45.4 tokens/second 100% In-VRAM on NVIDIA GeForce RTX 4050 6GB Laptop GPU (zero PCIe paging stalls).</li>
</ul>
<p>Open source repository: <a href="https://github.com/AmSach/maple">https://github.com/AmSach/maple</a></p>
<p>Hugging Face Models: <a href="https://huggingface.co/AmSach/Gemma-4-26B-A4B-MAPLE-Sub5GB-GGUF">Gemma 4 Sub-5GB</a></p>
"""

def update_zenodo(token):
    headers = {"Authorization": f"Bearer {token}"}
    base_url = "https://zenodo.org/api"

    print(f"[*] Accessing Zenodo deposition for record {RECORD_ID}...")
    # Step 1: Create a new version of the published deposition
    r = requests.post(f"{base_url}/deposit/depositions/{RECORD_ID}/actions/newversion", headers=headers)
    if r.status_code != 201:
        print(f"[!] Error creating new version: HTTP {r.status_code}")
        print(r.text)
        return False

    new_version_data = r.json()
    new_draft_url = new_version_data["links"]["latest_draft"]
    new_id = new_draft_url.split("/")[-1]
    print(f"[+] Created new draft deposition: ID {new_id}")

    # Fetch draft details
    r = requests.get(f"{base_url}/deposit/depositions/{new_id}", headers=headers)
    draft_data = r.json()

    # Step 2: Delete old files from the draft
    for f in draft_data.get("files", []):
        file_id = f["id"]
        print(f"[*] Deleting existing file {f['filename']} (ID: {file_id})...")
        requests.delete(f"{base_url}/deposit/depositions/{new_id}/files/{file_id}", headers=headers)

    # Step 3: Upload new compiled main.pdf
    print(f"[*] Uploading updated PDF: {PDF_PATH} ({os.path.getsize(PDF_PATH)} bytes)...")
    bucket_url = draft_data.get("links", {}).get("bucket")
    if bucket_url:
        with open(PDF_PATH, "rb") as fp:
            r = requests.put(f"{bucket_url}/main.pdf", data=fp, headers=headers)
    else:
        with open(PDF_PATH, "rb") as fp:
            r = requests.post(f"{base_url}/deposit/depositions/{new_id}/files",
                              data={"name": "main.pdf"},
                              files={"file": fp},
                              headers=headers)

    if r.status_code not in (200, 201):
        print(f"[!] Error uploading file: HTTP {r.status_code}")
        print(r.text)
        return False
    print("[+] PDF file uploaded successfully.")

    # Step 4: Update metadata
    print("[*] Updating deposition metadata with canonical benchmarks...")
    metadata = draft_data.get("metadata", {})
    metadata.update({
        "title": "MAPLE: Manifold-Activated Pruning of Latent Experts - Empirical Activation-Guided Expert Distillation for Mixture-of-Experts Language Models Below 2 Bits Per Weight",
        "upload_type": "publication",
        "publication_type": "preprint",
        "description": DESCRIPTION,
        "keywords": [
            "Mixture of Experts",
            "MoE Pruning",
            "Gemma 4",
            "Model Compression",
            "Sub-2-BPW",
            "GGUF",
            "llama.cpp",
            "Edge AI",
            "HumanEval",
            "MBPP",
            "GSM8K"
        ]
    })

    r = requests.put(f"{base_url}/deposit/depositions/{new_id}",
                     data=json.dumps({"metadata": metadata}),
                     headers={"Content-Type": "application/json", **headers})
    if r.status_code != 200:
        print(f"[!] Error updating metadata: HTTP {r.status_code}")
        print(r.text)
        return False
    print("[+] Metadata updated successfully.")

    # Step 5: Publish the new version
    print("[*] Publishing new version...")
    r = requests.post(f"{base_url}/deposit/depositions/{new_id}/actions/publish", headers=headers)
    if r.status_code != 202:
        print(f"[!] Error publishing: HTTP {r.status_code}")
        print(r.text)
        return False

    publish_data = r.json()
    doi = publish_data.get("doi")
    print("\n" + "=" * 80)
    print("  ZENODO PREPRINT UPDATE COMPLETE!")
    print(f"  New Version DOI: {doi}")
    print(f"  Record URL: https://zenodo.org/records/{new_id}")
    print("=" * 80)
    return True

if __name__ == "__main__":
    token = os.environ.get("ZENODO_TOKEN")
    if len(sys.argv) > 1:
        token = sys.argv[1]

    if not token:
        print("[!] Zenodo API token required.")
        print("    Usage: python update_zenodo.py <YOUR_ZENODO_TOKEN>")
        print("    Or set the ZENODO_TOKEN environment variable.")
        sys.exit(1)

    update_zenodo(token)
