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
