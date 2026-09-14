"""Local human-review UI for additive CubiCasa room-graph corrections."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas


ROOM_TYPES = ["LivingRoom", "Bedroom", "Kitchen", "Dining", "Bath", "Storage", "Entry", "Garage", "Other", "Outdoor"]
EDGE_TYPES = ["adjacent_to", "connected_by_door"]
COLOURS = ["#ed174c", "#00a8f3", "#ffc400", "#ff7b67", "#44e0bb", "#668b1b", "#ff5ca8", "#8b5a2b", "#c95b99", "#55cc00"]


def read_records(uploaded) -> list[dict]:
    return [json.loads(line) for line in uploaded.getvalue().decode("utf-8").splitlines() if line.strip()]


def canvas_json(rooms: list[dict], edges: list[dict], image_size: tuple[int, int], display_width: int) -> dict:
    scale = min(1.0, display_width / image_size[0])
    boxes = [
        {"type": "rect", "left": room["bbox_xyxy"][0] * scale, "top": room["bbox_xyxy"][1] * scale,
         "width": (room["bbox_xyxy"][2] - room["bbox_xyxy"][0]) * scale,
         "height": (room["bbox_xyxy"][3] - room["bbox_xyxy"][1]) * scale,
         "fill": "transparent", "stroke": COLOURS[ROOM_TYPES.index(room["category_name"]) % len(COLOURS)],
         "strokeWidth": 3, "room_id": room["room_id"]}
        for room in rooms
    ]
    centres = {room["room_id"]: ((room["bbox_xyxy"][0] + room["bbox_xyxy"][2]) * scale / 2, (room["bbox_xyxy"][1] + room["bbox_xyxy"][3]) * scale / 2) for room in rooms}
    lines = [
        {"type": "line", "x1": centres[edge["room_a"]][0], "y1": centres[edge["room_a"]][1], "x2": centres[edge["room_b"]][0], "y2": centres[edge["room_b"]][1], "stroke": "#c026d3" if edge["predicate"] == "connected_by_door" else "#2563eb", "strokeWidth": 3, "strokeDashArray": [8, 6] if edge["predicate"] == "adjacent_to" else None, "selectable": False, "evented": False}
        for edge in edges if edge["room_a"] in centres and edge["room_b"] in centres
    ]
    return {"version": "4.4.0", "objects": lines + boxes}


def boxes_from_canvas(rooms: list[dict], canvas: dict | None, image_size: tuple[int, int], display_width: int) -> list[dict]:
    updated = deepcopy(rooms)
    if not canvas or not canvas.get("objects"):
        return updated
    scale = min(1.0, display_width / image_size[0])
    by_id = {str(item.get("room_id")): item for item in canvas["objects"] if item.get("room_id")}
    for position, room in enumerate(updated):
        item = by_id.get(room["room_id"])
        if item is None and position < len(canvas["objects"]):
            item = canvas["objects"][position]
        if item is None:
            continue
        x0, y0 = float(item.get("left", 0)) / scale, float(item.get("top", 0)) / scale
        x1 = x0 + float(item.get("width", 0)) * float(item.get("scaleX", 1)) / scale
        y1 = y0 + float(item.get("height", 0)) * float(item.get("scaleY", 1)) / scale
        room["bbox_xyxy"] = [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)]
    return updated


def validate(rooms: list[dict], edges: list[dict], image_size: tuple[int, int]) -> list[str]:
    errors, room_ids, seen = [], {room["room_id"] for room in rooms}, set()
    for room in rooms:
        x0, y0, x1, y1 = room["bbox_xyxy"]
        if not (0 <= x0 < x1 <= image_size[0] and 0 <= y0 < y1 <= image_size[1]):
            errors.append(f"Invalid box: {room['room_id']}")
    for edge in edges:
        key = tuple(sorted((edge["room_a"], edge["room_b"]))) + (edge["predicate"],)
        if edge["room_a"] == edge["room_b"] or edge["room_a"] not in room_ids or edge["room_b"] not in room_ids:
            errors.append(f"Invalid edge: {edge}")
        elif key in seen:
            errors.append(f"Duplicate edge: {edge}")
        seen.add(key)
    return errors


st.set_page_config(page_title="CubiCasa graph review", layout="wide")
st.title("CubiCasa room-graph reviewer")
st.caption("Edits create an additive review patch. They never overwrite the source SVG or CubiGraph silver graph.")

with st.sidebar:
    manifest_upload = st.file_uploader("Canonical JSONL review packet", type="jsonl")
    image_upload = st.file_uploader("Floor-plan image for selected plan", type=["png", "jpg", "jpeg"])
    reviewer = st.text_input("Reviewer ID", value="")

if not manifest_upload or not image_upload:
    st.info("Upload a small canonical JSONL packet and the selected plan image.")
    st.stop()
records = read_records(manifest_upload)
selected_id = st.selectbox("Plan", [record["plan_id"] for record in records])
record = next(record for record in records if record["plan_id"] == selected_id)
image = Image.open(image_upload).convert("RGB")
if list(image.size) != record["image_size"]:
    st.warning(f"Image is {image.size}; packet says {record['image_size']}. Use the matching image before exporting a review.")
image_size = image.size
display_width = min(1000, image.width)

if f"rooms:{selected_id}" not in st.session_state:
    st.session_state[f"rooms:{selected_id}"] = deepcopy(record["rooms"])
    st.session_state[f"edges:{selected_id}"] = [
        {"room_a": edge["room_a"], "room_b": edge["room_b"], "predicate": edge["predicate"]} for edge in record["silver_edges"]
    ]
rooms_key, edges_key = f"rooms:{selected_id}", f"edges:{selected_id}"

left, right = st.columns([3, 2])
with left:
    st.subheader("Drag or resize room boxes")
    canvas = st_canvas(fill_color="rgba(0,0,0,0)", stroke_width=3, background_image=image, initial_drawing=canvas_json(st.session_state[rooms_key], st.session_state[edges_key], image_size, display_width), drawing_mode="transform", width=display_width, height=round(image.height * min(1.0, display_width / image.width)), key=f"canvas:{selected_id}")
    if st.button("Apply box positions from canvas"):
        st.session_state[rooms_key] = boxes_from_canvas(st.session_state[rooms_key], canvas.json_data, image_size, display_width)
        st.rerun()
with right:
    st.subheader("Rooms")
    room_frame = pd.DataFrame([{**room, "bbox_xyxy": json.dumps(room["bbox_xyxy"])} for room in st.session_state[rooms_key]])
    edited_rooms = st.data_editor(room_frame, column_config={"category_name": st.column_config.SelectboxColumn(options=ROOM_TYPES), "room_id": st.column_config.TextColumn(disabled=True)}, hide_index=True, use_container_width=True, key=f"room-table:{selected_id}")
    if st.button("Apply room table"):
        rebuilt = []
        for row in edited_rooms.to_dict("records"):
            row["bbox_xyxy"] = json.loads(row["bbox_xyxy"])
            row["category_id"] = ROOM_TYPES.index(row["category_name"]) + 1
            rebuilt.append(row)
        st.session_state[rooms_key] = rebuilt
        st.rerun()
    with st.expander("Add room"):
        new_id = st.text_input("New room ID", key=f"new-id:{selected_id}")
        new_type = st.selectbox("New room type", ROOM_TYPES, key=f"new-type:{selected_id}")
        new_box = st.text_input("Box [x0, y0, x1, y1]", value="[0, 0, 100, 100]", key=f"new-box:{selected_id}")
        if st.button("Add room"):
            st.session_state[rooms_key].append({"room_id": new_id, "category_id": ROOM_TYPES.index(new_type) + 1, "category_name": new_type, "bbox_xyxy": json.loads(new_box)})
            st.rerun()

st.subheader("Graph edges")
edge_frame = pd.DataFrame(st.session_state[edges_key], columns=["room_a", "room_b", "predicate"])
edited_edges = st.data_editor(edge_frame, column_config={"room_a": st.column_config.SelectboxColumn(options=[room["room_id"] for room in st.session_state[rooms_key]]), "room_b": st.column_config.SelectboxColumn(options=[room["room_id"] for room in st.session_state[rooms_key]]), "predicate": st.column_config.SelectboxColumn(options=EDGE_TYPES)}, num_rows="dynamic", hide_index=True, use_container_width=True, key=f"edge-table:{selected_id}")
if st.button("Apply edge table"):
    st.session_state[edges_key] = edited_edges.to_dict("records")
    st.rerun()

notes = st.text_area("Reviewer notes / reasons")
errors = validate(st.session_state[rooms_key], st.session_state[edges_key], image_size)
if errors:
    st.error("Fix before export: " + "; ".join(errors))
else:
    patch = {"plan_id": selected_id, "reviewer": reviewer, "source_graph_provenance": record["provenance"].get("graph_provenance", "silver"), "review_status": "reviewed", "rooms": st.session_state[rooms_key], "edges": st.session_state[edges_key], "reviewer_notes": notes}
    st.success("Valid review record ready.")
    st.download_button("Download reviewed graph JSON", json.dumps(patch, indent=2), file_name=f"{selected_id}_review.json", mime="application/json")
