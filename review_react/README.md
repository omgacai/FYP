# CubiCasa React graph reviewer

Local browser reviewer for CubiGraph relation SVGs. It preserves the source SVG diagram and exports an additive JSON correction containing room labels, boxes, relation edits, reviewer ID, and notes.

## Start

```bash
cd /Users/numkoos/Downloads/FYP/review_react
npm install --offline
npm run dev
```

Open the localhost URL printed by Vite. Upload the matching relation SVG and floor-plan image. The overlay uses the SVG's own coordinate system at a uniform width scale, avoiding the vertical stretching that broke the Streamlit version.

## Review workflow

1. Click a coloured room region and correct its semantic label.
2. Adjust its source-SVG bounding box only if needed.
3. Remove incorrect relation edges or add a new adjacency/door relation.
4. Record a short reason and download the correction JSON.

The exporter produces a review patch; it never overwrites CubiCasa source annotations or the silver CubiGraph result.

## Manual raster benchmark mode (default)

Open the app without query parameters to annotate directly from a PNG/JPG; no SVG is needed. Place nodes, switch to Move/select to drag or rename them, then Connect nodes to add typed undirected edges. Edge types distinguish doors, open passages, shared-wall-only adjacency and uncertainty. Select a room in the sidebar to edit or delete it. Undo restores the last graph edit (up to 50).

Set a unique source-based plan ID (CubiCasa images often share filenames) and annotator ID. Check every room pair, mark ambiguous pairs uncertain, tick the completeness statement, and mark reviewed. Download JSON after marking reviewed. Coordinates are normalized to the original raster. Image dimensions and SHA-256 bind annotations to their source image.

Download JSON at any point for a draft. To resume, reopen the JSON and select its original image; mismatched image bytes are rejected. The browser retains one draft as recovery, without the image. Downloaded files are the durable record. New plan clears the working canvas; save the current JSON first. Nothing is sent to Drive or a model.

The original SVG workflow remains at `/?mode=review`. The active experiment checklist is in `../GRAPH_QA_BENCHMARK_PLAN.md`.


### Room outlines and derived boxes

Choose **Draw room outline**, click the corners in boundary order, then **Finish outline**. The final edge closes automatically; do not repeat the first corner. Concave rooms (including L shapes) are supported. Trace the room’s interior boundary consistently. This is a single polygon per room; holes or disconnected regions are not supported in this version.

To add geometry to a previously placed node, select it and choose **Draw outline for selected room**. To refine a shape, use Move/select and drag its white corner handles, or open **Edit outline corners** to insert/remove corners. Redraw replaces the shape while keeping the room ID and connections. Undo restores the previous edit. Crossing or zero-area polygons are rejected. A new node is placed inside its polygon; moving a node does not move the room boundary.

The dashed rectangle is the axis-aligned enclosing box, recalculated from the polygon. Toggle **Show derived boxes** to hide it. A concave room’s box can cover non-room space; this is expected and does not create graph edges.

JSON schema `floorplan-manual-graph/2` stores `polygon: [[x,y],…]` and derived `bbox_xyxy: [xmin,ymin,xmax,ymax]` on each node, in normalized original-image coordinates. Multiply x coordinates by image width and y coordinates by height for pixel coordinates. COCO conversion uses `[xmin*W, ymin*H, (xmax-xmin)*W, (ymax-ymin)*H]`; keep polygons separately. The EGTR dataset exporter still needs an adapter for this schema and reviewed labels.

Version-1 point-only JSON imports are supported; missing geometry remains null. All rooms require outlines to mark a new record reviewed. Draft downloads remain possible without all outlines. Finish an in-progress outline before downloading; unfinished clicks are not included in browser autosave. Saved outlines, boxes, nodes and edges are included. Save locations remain browser draft storage and downloaded JSON files.


### Automatic local folder saving

When running `npm run dev` (or `npm run preview`), uploading a PNG/JPG now copies its bytes into `FYP/cubicasa_eval/images/`. Graph changes autosave after a short pause to the matching JSON in `FYP/cubicasa_eval/annotations/`. Wait for **Saved automatically** before closing the page. Finish an outline to include it in the saved graph; unfinished corner clicks remain transient. Download JSON remains a fallback if folder saving fails.

Names combine the source directory when recognized from the exclusion manifest with the full SHA-256 image hash. Unregistered images use `plan--<hash>`; browsers do not reveal the original folder path. Different images with identical filenames stay separate. Re-uploading identical bytes reuses the stored image and reopens its annotation. A second tab with an outdated revision is blocked from overwriting newer work. Image copying does not itself add an entry to the training exclusion manifest.

This feature is served by the local Vite middleware, not a cloud service. Static hosting without that middleware cannot save to the folder. The entire `cubicasa_eval` directory remains gitignored. The older browser-only saving instructions above describe the original workflow; local folder autosaving now supplements browser recovery and manual downloads.
