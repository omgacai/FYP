from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from retrieval_app.data.manifest import load_manifest
from retrieval_app.representations.graph import graph_summary
from retrieval_app.representations.query_parser import parse_query
from retrieval_app.representations.visual import ClipVisualTextEncoder
from retrieval_app.services.retriever import FusionWeights, retrieve


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "cubicasa5k" / "manifests" / "cubicasa_image_svg.jsonl"


@st.cache_resource(show_spinner=False)
def load_clip() -> ClipVisualTextEncoder:
    return ClipVisualTextEncoder()


def _display_score(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


st.set_page_config(page_title="Floorplan Retrieval Lab", layout="wide")
st.title("Floorplan Retrieval Lab")
st.caption("Compare visual, room-graph, and geometry evidence before training a fusion model.")

with st.sidebar:
    st.header("Corpus")
    manifest_value = st.text_input("Manifest JSONL path", value=str(DEFAULT_MANIFEST))
    root_value = st.text_input("Corpus root", value="", help="Optional. Relative paths resolve from this folder.")
    st.caption("The manifest builder creates image/SVG entries first. Add graph_path, room_counts, and provenance after graph extraction.")
    st.divider()
    st.header("Fusion weights")
    visual_weight = st.slider("Visual / CLIP", 0.0, 1.0, 0.40, 0.05)
    graph_weight = st.slider("Room graph", 0.0, 1.0, 0.35, 0.05)
    geometry_weight = st.slider("Geometry / counts", 0.0, 1.0, 0.25, 0.05)
    use_clip = st.checkbox("Enable CLIP ViT-B/32", value=False, help="Downloads/loads OpenCLIP weights on first use.")

manifest_path = Path(manifest_value).expanduser()
if not manifest_path.exists():
    st.info(
        "Create a corpus manifest first. For CubiCasa5K: "
        "`python -m retrieval_app.scripts.build_cubicasa_manifest "
        "--cubicasa-root /path/to/cubicasa5k --output /path/to/manifests/cubicasa_image_svg.jsonl --limit 20`"
    )
    st.stop()

try:
    plans = load_manifest(manifest_path, Path(root_value) if root_value else None)
except Exception as error:
    st.exception(error)
    st.stop()

st.success(f"Loaded {len(plans)} plans from {manifest_path.name}")
query_text = st.text_area(
    "Buyer query",
    value="I need 3 bedrooms with a kitchen next to the dining room.",
    height=100,
)
query = parse_query(query_text)

left, right = st.columns((1, 2))
with left:
    st.subheader("Structured query baseline")
    st.caption("This transparent parser is deliberately conservative; replace it later with an evaluated LLM-to-JSON module.")
    st.json({
        "room_counts": query.room_counts,
        "relations": [item.__dict__ for item in query.relations],
        "requires_verified_connectivity": query.requires_verified_connectivity,
    })
with right:
    st.subheader("Experiment contract")
    st.code(
        "score = w_visual × CLIP(query, image) + "
        "w_graph × typed_relation_match + w_geometry × count_match",
        language="text",
    )
    if query.requires_verified_connectivity:
        st.warning("This query needs connectivity evidence. Interpret graph scores only for plans marked `gold` or manually audited.")

if st.button("Rank plans", type="primary"):
    try:
        scorer = load_clip() if use_clip else None
        with st.spinner("Scoring corpus…"):
            results = retrieve(
                query,
                plans,
                FusionWeights(visual_weight, graph_weight, geometry_weight),
                visual_scorer=scorer,
            )
    except Exception as error:
        st.exception(error)
        st.stop()

    table = pd.DataFrame([
        {
            "rank": index + 1,
            "plan_id": item.plan.plan_id,
            "fused": round(item.fused_score, 3),
            "visual": _display_score(item.scores.visual),
            "graph": _display_score(item.scores.graph),
            "geometry": _display_score(item.scores.geometry),
            "graph provenance": item.plan.graph_provenance,
        }
        for index, item in enumerate(results)
    ])
    st.subheader("Ranked results")
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.download_button("Download rankings CSV", table.to_csv(index=False), "retrieval_rankings.csv", "text/csv")

    st.subheader("Inspect top results")
    for index, item in enumerate(results[:6], start=1):
        with st.expander(f"#{index} · {item.plan.plan_id} · fused {item.fused_score:.3f}", expanded=index == 1):
            image_column, details_column = st.columns((1, 2))
            with image_column:
                if item.plan.image_path.exists():
                    st.image(Image.open(item.plan.image_path), caption=item.plan.image_path.name, use_container_width=True)
                else:
                    st.error(f"Image missing: {item.plan.image_path}")
            with details_column:
                st.write(item.plan.description or "No plan description in manifest.")
                st.json({
                    "room_counts": item.plan.room_counts,
                    "geometry": item.plan.geometry,
                    "graph_summary": graph_summary(item.plan),
                    "graph_provenance": item.plan.graph_provenance,
                })
                st.markdown("**Constraint evidence**")
                for line in item.evidence:
                    st.write(f"- {line}")
                if item.plan.svg_path and item.plan.svg_path.exists():
                    st.download_button(
                        "Download source SVG",
                        item.plan.svg_path.read_text(encoding="utf-8"),
                        file_name=f"{item.plan.plan_id}.svg",
                        mime="image/svg+xml",
                        key=f"svg-{item.plan.plan_id}",
                    )
