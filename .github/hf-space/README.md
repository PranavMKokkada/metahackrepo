---
title: Autonomous SRE
emoji: "🛠️"
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Autonomous SRE Space

**Read the human write-up (environment, training arc, rewards, HF story):**  
### [**BLOG_POST.md**](./BLOG_POST.md)

This Space is automatically synced from the GitHub repository using the
`Deploy Hugging Face Space` workflow.

The app runs via `Dockerfile` and serves the API/UI on port `7860`.

## HTTPS / Gradio on Spaces

The public URL is **HTTPS**, but the container sees **HTTP** on port 7860. The app must trust `X-Forwarded-Proto` (Uvicorn `proxy_headers=True` and `forwarded_allow_ips`) so Gradio generates **https://** API and asset URLs. If that is wrong, the browser reports mixed content, `503`, `ERR_ADDRESS_UNREACHABLE`, or “Unsafe attempt to load URL http://… from https://…”.

## Training notebook & blog (public files in this repo)

- **Training run notebook:** [CodeOrganismVM_Training.ipynb](./CodeOrganismVM_Training.ipynb)  
- **Blog post (markdown):** [BLOG_POST.md](./BLOG_POST.md)  

Open both in an **Incognito** window to confirm they load without logging in.
