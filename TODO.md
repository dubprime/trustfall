# TODO

## 1. Code Organization & Refactoring
- [ ] Consolidate duplicate scripts: Merge the various versions (pie_matrix_v2.py, v3, v4, v5, v6, pie_matrix.py, default_matrix.py) into a single, well-documented module. _(In progress: v7 is the new base, old files remain for consolidation)_
- [x] Modularize core logic: Extract simulation, plotting, and input-handling logic into reusable modules or packages.
- [x] Rename files for clarity and consistency.

## 2. Testing & Quality
- [ ] Add automated tests: Implement unit and integration tests for simulation logic and data validation (e.g., with pytest).
- [ ] Set up CI: Integrate with GitHub Actions or another CI tool to run tests and lint checks on every commit.
- [ ] Add type checking: Use mypy or similar tools to enforce type hints and catch type errors early.

## 3. Documentation
- [ ] Expand README.md: Add usage instructions, feature overview, input/output descriptions, and example screenshots.
- [x] Document code: Add docstrings to all functions and classes, and ensure parameter/return types are clear.
- [ ] Add a changelog: Track major changes and improvements over time.

## 4. User Experience & Features
- [ ] Improve input validation: Ensure all user inputs in Streamlit are validated (e.g., allocations sum to 100%, no negative values).
- [ ] Add scenario saving/loading: Allow users to save and load parameter sets or scenarios.
- [ ] Export results: Add options to export simulation results and charts as CSV, Excel, or images.
- [ ] Enhance visualizations: Add tooltips, better legends, and interactivity to charts for improved clarity.
- [ ] Accessibility: Ensure the app is usable for all users (colorblind-friendly palettes, keyboard navigation, etc.).

## 5. Performance & Robustness
- [ ] Profile and optimize simulation: For large numbers of months or scenarios, ensure the simulation runs efficiently.
- [x] Handle missing dependencies gracefully: Provide clear error messages and installation instructions if required packages are missing.

## 6. Dependency Management
- [ ] Pin dependency versions: Specify exact versions in requirements.txt to ensure reproducibility.
- [ ] Add optional dev dependencies: Include tools like pytest, mypy, and black in a dev-requirements.txt.

## 7. Advanced/Stretch Goals
- [ ] Parameter sensitivity analysis: Add tools to analyze how outputs change with different input parameters.
- [ ] Batch scenario comparison: Allow users to run and compare multiple scenarios side-by-side.
- [ ] API support: Expose simulation logic as a REST API for programmatic access. 