# Product Requirements Document (PRD)
**Project Name:** Multilingual Tourism Review Intelligence Engine
**Codename:** cluster-sentiment-engine
**Target Build Environment:** Google Antigravity / Cursor / Claude Code
**Author:** [Your Name]
**Status:** Draft / Ready for Build

---

## 1. Executive Summary

Tourism boards and hotel cluster managers in India currently rely on aggregate star ratings (e.g., "4.1★") that hide critical operational failures. A property can maintain a strong overall score while its food quality or cleanliness sentiment collapses, masked by high scores in location or staff. Furthermore, most review analytics tools fail to process **Hinglish (Hindi-English code-mixed) and Devanagari** text, which constitutes a majority of user-generated content from Indian tourists.

This project builds a **cluster-level review analytics engine** that transforms raw, multilingual tourist reviews into **aspect-level, competitive, and time-aware intelligence**. It answers a single, high-stakes question for a property manager: *"On which aspect am I behind my cluster, since when, and is it just me or the whole market?"*

This is not a sentiment dashboard. It is a **decision-support tool** for competitive benchmarking and market-wide issue detection.

---

## 2. Problem Statement & Business Context

### 2.1 The Core Problem
A single average rating is a vanity metric. It hides:
- **Aspect-level failures:** A property at 4.1★ may have a 2.8★ food sentiment.
- **Market-wide vs. property-specific issues:** A winter dip in "Room" sentiment might be a destination-level problem (e.g., all hotels lack adequate heating), not a competitive gap for one property.
- **Multilingual blind spots:** Existing tools cannot parse Hinglish, Devanagari, or mixed-language reviews, which contain the most nuanced feedback.

### 2.2 Why This Matters (The "So What")
- **Property Managers:** They need to know where to invest (e.g., retrain kitchen staff vs. upgrade room heaters) based on whether a decline is *theirs* or *everyone's*.
- **Tourism Boards / DMOs:** They need to identify destination-level issues that move together across a cluster (e.g., seasonal transport complaints) to target interventions.
- **Recruiters:** This project demonstrates end-to-end capability in **multilingual NLP, aspect-based sentiment analysis, competitive benchmarking, and decision-support system design**—skills directly applicable to roles in product analytics, consulting, and hospitality tech.

---

## 3. Goals & Non-Goals

### 3.1 Goals
- Build a modular, reproducible pipeline that ingests raw multilingual reviews and outputs aspect-level sentiment scores.
- Implement a **rule/lexicon-based classifier** for 8 core aspects that is transparent, explainable, and handles Hinglish/Devanagari.
- Generate a **competitive benchmark** comparing each property against its cluster mean, with clear **lead/lag flags**.
- Classify a sentiment decline as **market-wide** or **property-specific**.
- Detect **changepoints** (when a problem started) per property-aspect.
- Deliver a **Streamlit dashboard** for interactive exploration by non-technical stakeholders.
- Include **tests that recover injected patterns** from synthetic data to validate the pipeline.

### 3.2 Non-Goals (for MVP)
- Fine-tuning a large multilingual transformer (mBERT, XLM-R) — this is a roadmap item.
- Scraping live data from OTAs (TripAdvisor, Booking.com) — use synthetic data or public datasets for the MVP.
- Building a full production web application with authentication and role-based access.
- Handling languages beyond Hinglish, Devanagari, and English in the MVP.

---

## 4. User Personas & Stories

| Persona | Role | Key Question | User Story |
| :--- | :--- | :--- | :--- |
| **Ravi** | Property Manager, Mussoorie | "Is my food problem my fault or the market's?" | As Ravi, I want to see my property's aspect scores vs. the cluster mean so I know where to invest. |
| **Priya** | Tourism Board Analyst, Uttarakhand | "Which issues are destination-wide?" | As Priya, I want to see which aspects move together across all properties so I can target policy interventions. |
| **Arjun** | Data Analyst (Hiring Manager) | "Can this person build a real decision tool?" | As Arjun, I want to see a project that goes beyond dashboards and provides actionable, business-ready recommendations. |

---

## 5. Functional Requirements

### FR-1: Data Ingestion & Schema
- **FR-1.1:** The system shall load reviews from CSV and JSON files.
- **FR-1.2:** The schema shall include: `review_id`, `property_id`, `cluster_id`, `review_text`, `rating`, `review_date`, `reviewer_name` (optional), `stay_date` (optional).
- **FR-1.3:** **PII shall be dropped at ingestion** (e.g., reviewer name, email) to demonstrate privacy awareness.
- **FR-1.4:** A synthetic data generator shall produce 12 properties across 3 clusters with **injected patterns** (e.g., a specific property's "Food" sentiment declining over time; a cluster-wide "Room" issue in winter).

### FR-2: Text Normalization
- **FR-2.1:** The system shall normalize text for **English, Hinglish (Romanized Hindi), and Devanagari**.
- **FR-2.2:** Normalization shall handle: emoji removal/translation, negation handling (e.g., "not good" → "bad"), and basic spelling correction for common Hinglish variations.
- **FR-2.3:** The system shall output a cleaned `normalized_text` field for downstream classification.

### FR-3: Aspect Taxonomy & Classification
- **FR-3.1:** The system shall classify reviews into **8 aspects**: Food, Room, Housekeeping, Staff, Location, Value for Money, Cleanliness, Booking Experience.
- **FR-3.2:** The classifier shall be **rule/lexicon-based** for the MVP, using multilingual cue phrases per aspect.
- **FR-3.3:** The classifier shall output an **aspect-sentiment score** (positive, negative, neutral) per aspect per review.
- **FR-3.4:** The classifier shall be **pluggable** via a `Classifier` interface, allowing future model backends (e.g., fine-tuned transformers) to be swapped in without changing downstream logic.

### FR-4: Competitive Benchmarking
- **FR-4.1:** The system shall compute the **cluster mean** for each aspect.
- **FR-4.2:** The system shall generate **lead/lag flags** for each property-aspect (e.g., "Lead", "Lag", "Neutral").
- **FR-4.3:** The system shall **classify a sentiment decline as market-wide or property-specific** by comparing a property's trend to the cluster trend.

### FR-5: Trend & Changepoint Detection
- **FR-5.1:** The system shall compute a monthly (or weekly) sentiment time series per property-aspect.
- **FR-5.2:** The system shall detect **simple changepoints** (e.g., using a basic CUSUM or moving-average threshold) to identify when a sentiment shift occurred.
- **FR-5.3:** The output shall include the **changepoint date** and the **direction** (improvement/decline) per property-aspect.

### FR-6: Output & Visualization
- **FR-6.1:** The system shall generate a **Streamlit dashboard** with the following views:
    - **Cluster Overview:** Heatmap of aspect scores by property.
    - **Property Deep-Dive:** Aspect trend lines, lead/lag flags, and changepoint markers.
    - **Market-Wide vs. Property-Specific:** A view that highlights issues affecting multiple properties.
- **FR-6.2:** The dashboard shall allow filtering by cluster, property, aspect, and date range.
- **FR-6.3:** The system shall export a **one-page "Action Report"** per property, summarizing key findings and recommendations.

### FR-7: Testing & Validation
- **FR-7.1:** The system shall include **unit tests** for the normalizer, classifier, and benchmarker.
- **FR-7.2:** The system shall include **integration tests** that recover the injected patterns from the synthetic data (e.g., confirming that the declining "Food" sentiment property is flagged as a laggard).
- **FR-7.3:** The system shall include a **reproducible figures script** (`scripts/make_figures.py`) that regenerates all dashboard figures from the synthetic data.

---

## 6. Non-Functional Requirements

- **NFR-1: Reproducibility:** The entire pipeline must be reproducible from a single command (`make all` or `python run_pipeline.py`).
- **NFR-2: Explainability:** The classifier's decisions must be explainable (e.g., which cue phrases triggered an aspect-sentiment score).
- **NFR-3: Modularity:** Each component (loader, normalizer, classifier, benchmarker, changepoint detector) must be a separate module with a clean interface.
- **NFR-4: Performance:** The pipeline shall process 10,000 reviews in under 60 seconds on a standard laptop.
- **NFR-5: Code Quality:** The code must follow PEP 8, use type hints, and include docstrings for public functions. **Comments shall be sparse and purposeful** (explain *why*, not *what*).

---

## 7. Technical Architecture & Stack

### 7.1 High-Level Architecture
