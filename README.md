# LivestockLens — Philippine Livestock Production Analytics

This is the complete GitHub-ready source package for the public Streamlit application.

## Folder layout

```text
.streamlit/
  config.toml                 # Light theme and forest-green sidebar
livestock_analytics/
  streamlit_app.py            # Main responsive Streamlit UI
  app.py                      # Existing data preparation and ML backend (keep this)
  requirements.txt            # Streamlit, Plotly and ML dependencies
  notebooks/
    LivestockLens_Final_Colab_Analysis.ipynb
  data/
  static/
  templates/
```

## Deploy using Streamlit Community Cloud

1. Keep `.streamlit/` at the repository root and keep the `livestock_analytics/` folder beside it.
2. Connect this GitHub repository to Streamlit Community Cloud.
3. Set the main file path to `livestock_analytics/streamlit_app.py`.
4. The dependencies are declared in `livestock_analytics/requirements.txt`.

The final Colab notebook and the original Flask implementation are retained. The Streamlit app imports `livestock_analytics/app.py` for the shared dataset cleaning and ML workflow, so do not delete that file.

See `livestock_analytics/STREAMLIT_DEPLOYMENT.md` for detailed instructions and methodology notes.
