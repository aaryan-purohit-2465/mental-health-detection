# Dataset Plan

This project uses **publicly available, anonymised, or synthetic datasets only**.  
No real user posts or personal data will ever be uploaded to this repository.

## Datasets Considered

### 1. CLPsych Dataset (Reddit)
- Contains mental health–related posts.
- **Requires data request/access**, cannot upload here.
- Will use only small **synthetic** subsets inside `data/samples/`.

### 2. Dreaddit Dataset
- Public dataset for stress/depression detection.
- Will NOT store raw original posts here.  
- Instead, will load externally or generate small anonymised samples.

### 3. Synthetic Data (Safe to include)
- Short text examples written manually.
- Used for testing and demo purposes.
- Stored in `data/samples/`.

## Important Notes
- **This repo will never contain personal information.**
- All datasets must be anonymised before use.
- For training on full datasets, users must download data separately following original licenses.

## Files in this Directory
- `samples/` — contains small synthetic, anonymised text examples.
- `README.md` — this file.
