# Climate Analog Explorer — Streamlit deployment

This directory is generated from the research project by
`scripts/create_streamlit_deploy.py`.

Run locally from this directory:

    python -m pip install -r requirements.txt
    streamlit run app.py

Large `.h5` and `.npy` files are configured for Git LFS in `.gitattributes`.

The deployment databases intentionally omit `pooled384`, `row_off`, and
`col_off`. The original research databases are unchanged.
