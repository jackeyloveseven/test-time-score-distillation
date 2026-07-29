# Model files

Run `bash scripts/download_models.sh core` from the repository root. Paths:

```text
klein9b/       # required generator
klein9b-base/  # required frozen teacher
klein4b-base/  # optional legacy fake critic (`all` mode)
clip/          # optional CLIP scorer (`all` mode)
```

Large model files are intentionally ignored by Git.
