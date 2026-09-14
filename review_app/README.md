# CubiCasa graph reviewer

This local Streamlit MVP accepts one small canonical JSONL review packet and the
matching plan image. It supports box drag/resize, room-type edits, room addition,
edge edits, validation, and additive JSON export.

```bash
python -m pip install -r requirements_reviewer.txt
streamlit run review_app/app.py
```

Use a 10–20 plan packet copied from SOC. Save one downloaded review JSON per plan;
do not overwrite the source SVG or silver manifest.
