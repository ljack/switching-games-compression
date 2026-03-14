# Task 05: Auto-Load Original for Demo Comparison

## Status: completed

## Description
When user clicks "Load Demo", automatically also load demo-original.jpg in the background
and enable the comparison mode. This makes the demo more impressive by immediately showing
SWG3 vs JPEG side-by-side.

## Steps
1. In `loadDemo()`, after SWG decode completes, fetch demo-original.jpg in background
2. Set it as `originalImage` for comparison
3. Auto-activate the comparison bar with "2-up" mode as default
4. Show file size comparison: "SWG3: 6044 KB vs JPEG: 6800 KB" in the info bar
5. Handle case where original fails to load (network error) — just skip comparison

## Success Criteria
- Loading demo shows comparison immediately without extra clicks
- File size of both formats shown for easy comparison
- Graceful fallback if original image fails to load

## Files
- docs/app.js (loadDemo, comparison logic)
