# GitHub → Hugging Face Space (public links for submission)

## 1. Push this repository to GitHub

From your machine (with `git` and GitHub auth configured):

```bash
git add -A
git status
git commit -m "Wire UI to shared session; HF deploy payload; submission docs"
git push origin main
```

Wait for **CI** (`.github/workflows/ci.yml`) to pass.

## 2. Deploy workflow → Hugging Face

The workflow **Deploy Hugging Face Space** (`.github/workflows/deploy-hf-space.yml`) runs on pushes to **`main`**.

**GitHub repository settings**

| Type | Name | Value |
|------|------|--------|
| Secret | `HF_TOKEN` | Hugging Face token with write access to the Space |
| Variable | `HF_SPACE_REPO` | e.g. `your-org/your-space-name` (no `spaces/` prefix) |
| Variable | `HF_SPACE_BRANCH` | optional; default `main` |

After a green run, the Space git repo contains the same files as the Docker build context **plus** `BLOG_POST.md` and `CodeOrganismVM_Training.ipynb`.

## 3. Make the Space public

1. Open [https://huggingface.co/spaces](https://huggingface.co/spaces) and select your Space.  
2. **Settings** → **Change Space visibility** → **Public**.  
3. Save.

## 4. Submission URLs (replace `ORG/SPACE` with `HF_SPACE_REPO`)

On the Space **Files** tab, the story is **`BLOG_POST.md`** (synced from GitHub). The Space **README** also links it at the top so it is easy to find.

| What | URL |
|------|-----|
| **Live demo** | `https://huggingface.co/spaces/ORG/SPACE` |
| **Training notebook (blob)** | `https://huggingface.co/spaces/ORG/SPACE/blob/main/CodeOrganismVM_Training.ipynb` |
| **Blog post (blob)** | `https://huggingface.co/spaces/ORG/SPACE/blob/main/BLOG_POST.md` |
| **Raw notebook** | `https://huggingface.co/spaces/ORG/SPACE/raw/main/CodeOrganismVM_Training.ipynb` |

Verify each in an **Incognito** window (no login).

## 5. If deploy fails

- Confirm `HF_TOKEN` is not expired and has **write** on the Space.  
- Confirm `HF_SPACE_REPO` matches the Space id exactly.  
- Large binaries in GitHub are **not** copied by the whitelist deploy (by design); keep the Space payload small.
