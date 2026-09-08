# CLAUDE.md — Claude Code (CLI) Specific Guidelines

> **IMPORTANT IDENTITY NOTE:** 
> This configuration is specifically for **Claude Code (Anthropic's CLI agent running in the terminal)**. 
> It is distinct from Claude models running inside Antigravity IDE or Cursor. 
> You operate directly via terminal tools, bash execution, file read/writes, and git.

## Role & Mission
You are assisting Person 2 in a 36-hour acute ischemic stroke hackathon.
- **Project:** Automated ASPECTS Stroke Decision Support.
- **Your Scope as CLI Agent:** Fast terminal operations, dependency installs, test execution, script running, and git version control.
- **Person A Decoupling:** All current work runs against the synthetic dataset in `dummy_data/`. Do NOT wait for Person A.

## Project Structure
- `aspects_engine.py`: Bounding box extraction, normalized polygons, overlap scoring.
- `app.py`: Streamlit frontend.
- `generate_dummy_data.py`: Creates synthetic slices and stroke masks.
- `tune_polygons.py`: Visual preview of polygon fit.
- `TASKS.md`: Master task checklist.

## Quick CLI Commands
- Run dummy data generator: `python generate_dummy_data.py`
- Test polygon alignment: `python tune_polygons.py`
- Run Streamlit app: `streamlit run app.py`
- Check task status: Read `TASKS.md`

## Safety & Style Guidelines
1. **Never make clinical treatment claims:** Use observational phrasing (*"pattern consistent with..."*, *"requires clinician verification"*).
2. **Preserve dummy fallbacks:** Always keep synthetic data fallbacks active in `app.py`.
