"""
app.py
================================================================================
OSINT Investigative Agent — Streamlit application.

Pipeline: plan -> search -> scrape -> extract evidence -> verify -> answer.
Powered by Google's Gemini API (free tier via https://aistudio.google.com/apikey).
================================================================================
"""

from __future__ import annotations

import os
from typing import List, Optional

import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from pydantic import BaseModel, Field

from agent.utils import Settings, get_logger
from agent.planner import InvestigationPlanner, InvestigationPlan, SubQuestion
from agent.search import WebSearcher, SearchResult
from agent.extractor import PageScraper, EvidenceExtractor, EvidenceItem, ScrapedPage
from agent.verifier import EvidenceVerifier, VerificationResult
from agent.answer import AnswerGenerator, FinalAnswer

logger = get_logger(__name__)

load_dotenv()

# --------------------------------------------------------------------------
# Section 1: Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="OSINT Investigative Agent",
    page_icon="🔎",
    layout="wide",
)


def inject_custom_css() -> None:
    """Apply the case-file dark theme: monospace labels, bordered panels,
    and a stamped confidence badge — styled like an investigation dossier
    rather than a generic dark+neon dashboard."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

        :root {
            --ink: #E6EDF3;
            --ink-dim: #8B98A5;
            --panel: #131A21;
            --panel-border: #223140;
            --accent: #2DD4BF;
            --accent-dim: #14403B;
            --amber: #F2B84B;
            --red: #EF5350;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Headings and labels use the mono face for a dossier/terminal feel */
        h1, h2, h3, h4,
        [data-testid="stSidebar"] h1 {
            font-family: 'IBM Plex Mono', monospace !important;
            letter-spacing: 0.02em;
        }

        h1 {
            text-transform: uppercase;
            font-weight: 700 !important;
            border-bottom: 2px solid var(--accent);
            padding-bottom: 0.5rem;
            display: inline-block;
        }

        /* Sidebar styled like a case-file panel */
        [data-testid="stSidebar"] {
            background-color: var(--panel);
            border-right: 1px solid var(--panel-border);
        }
        [data-testid="stSidebar"] .stMarkdown p,
        [data-testid="stSidebar"] .stCaption {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.82rem;
        }
        [data-testid="stSidebar"] code {
            color: var(--accent);
            background-color: var(--accent-dim);
            border: 1px solid var(--panel-border);
            border-radius: 2px;
        }

        /* Rectangular, bordered buttons instead of soft rounded defaults */
        .stButton > button {
            font-family: 'IBM Plex Mono', monospace;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-size: 0.78rem;
            border: 1px solid var(--panel-border);
            border-radius: 3px;
            background-color: transparent;
            color: var(--ink);
            transition: border-color 0.15s ease, color 0.15s ease;
        }
        .stButton > button:hover {
            border-color: var(--accent);
            color: var(--accent);
        }

        /* Chat bubbles as case-note cards, square corners, left accent rule */
        [data-testid="stChatMessage"] {
            background-color: var(--panel);
            border: 1px solid var(--panel-border);
            border-left: 3px solid var(--accent);
            border-radius: 4px;
            padding: 0.25rem 0.5rem;
        }

        /* Expanders as bordered dossier sub-sections */
        [data-testid="stExpander"] {
            border: 1px solid var(--panel-border) !important;
            border-radius: 4px !important;
            background-color: rgba(255,255,255,0.015);
        }
        [data-testid="stExpander"] summary {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.85rem;
        }

        /* Status/progress log rendered like a terminal readout */
        [data-testid="stStatusWidget"],
        [data-testid="stExpander"] div[data-testid="stMarkdownContainer"] p {
            font-family: 'IBM Plex Mono', monospace;
        }

        hr {
            border-color: var(--panel-border) !important;
        }

        /* ---------------- Motion system ---------------- */

        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(10px); }
            to   { opacity: 1; transform: translateY(0); }
        }

        @keyframes stampIn {
            0%   { opacity: 0; transform: rotate(-3deg) scale(1.9); }
            60%  { opacity: 1; transform: rotate(-3deg) scale(0.94); }
            100% { opacity: 1; transform: rotate(-3deg) scale(1); }
        }

        @keyframes scanSweep {
            0%   { background-position: -200% 0; }
            100% { background-position: 200% 0; }
        }

        @keyframes appFadeIn {
            from { opacity: 0; }
            to   { opacity: 1; }
        }

        .stApp {
            animation: appFadeIn 0.4s ease-out;
        }

        /* Chat turns arrive like case files placed on the desk */
        [data-testid="stChatMessage"] {
            animation: fadeInUp 0.35s ease-out;
        }

        /* Ambient scanning sweep under the main title — ties to the
           "always scanning for information" idea rather than decoration */
        h1 {
            position: relative;
            border-bottom: none !important;
            padding-bottom: 0.6rem;
        }
        h1::after {
            content: "";
            position: absolute;
            left: 0;
            bottom: 0;
            width: 100%;
            height: 2px;
            background: linear-gradient(
                90deg,
                var(--panel-border) 0%,
                var(--panel-border) 40%,
                var(--accent) 50%,
                var(--panel-border) 60%,
                var(--panel-border) 100%
            );
            background-size: 250% 100%;
            animation: scanSweep 4.5s linear infinite;
        }

        /* Confidence stamp badge — lands like a rubber stamp hitting paper */
        .confidence-stamp {
            display: inline-block;
            font-family: 'IBM Plex Mono', monospace;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.78rem;
            padding: 0.3rem 0.7rem;
            border: 2px solid var(--stamp-color, var(--accent));
            color: var(--stamp-color, var(--accent));
            border-radius: 3px;
            transform: rotate(-3deg);
            margin-left: 0.6rem;
            vertical-align: middle;
            animation: stampIn 0.5s cubic-bezier(0.2, 0.8, 0.2, 1);
        }

        /* Tactile buttons: lift on hover, press down on click */
        .stButton > button {
            transition: border-color 0.15s ease, color 0.15s ease,
                        transform 0.12s ease, box-shadow 0.15s ease;
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 2px 8px rgba(45, 212, 191, 0.15);
        }
        .stButton > button:active {
            transform: translateY(0px) scale(0.98);
        }

        /* Expander panels ease their border color in on hover, signalling interactivity */
        [data-testid="stExpander"] {
            transition: border-color 0.2s ease;
        }
        [data-testid="stExpander"]:hover {
            border-color: var(--accent) !important;
        }

        /* Live terminal-style investigation log lines */
        .log-line {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.85rem;
            color: var(--ink-dim);
            animation: fadeInUp 0.3s ease-out;
        }
        .log-line .log-prompt {
            color: var(--accent);
            margin-right: 0.4rem;
        }

        @media (prefers-reduced-motion: reduce) {
            .stApp,
            [data-testid="stChatMessage"],
            .confidence-stamp,
            .log-line,
            h1::after {
                animation: none !important;
            }
            .stButton > button:hover,
            .stButton > button:active {
                transform: none !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_custom_css()


# --------------------------------------------------------------------------
# Section 2: Data model for a completed investigation (for chat history)
# --------------------------------------------------------------------------
class InvestigationRecord(BaseModel):
    question: str = Field(...)
    plan: Optional[InvestigationPlan] = None
    evidence_items: List[EvidenceItem] = Field(default_factory=list)
    verification: Optional[VerificationResult] = None
    final_answer: Optional[FinalAnswer] = None


# --------------------------------------------------------------------------
# Section 3: Settings loader (cached)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_settings() -> Settings:
    return Settings.load()


@st.cache_resource(show_spinner=False)
def configure_gemini(api_key: str) -> bool:
    """Configure the global Gemini SDK client once and cache that it's done."""
    genai.configure(api_key=api_key)
    return True


# --------------------------------------------------------------------------
# Section 4: Pipeline orchestration
# --------------------------------------------------------------------------
class InvestigationPipeline:
    """Orchestrates the full OSINT investigation pipeline end-to-end."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.planner = InvestigationPlanner(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            max_subquestions=settings.max_subquestions,
        )
        self.searcher = WebSearcher(max_results=settings.max_search_results_per_subquestion)
        self.scraper = PageScraper(
            user_agent=settings.scraper_user_agent,
            timeout_seconds=settings.request_timeout_seconds,
            max_page_chars=settings.max_page_chars,
        )
        self.extractor = EvidenceExtractor(model=settings.gemini_model)
        self.verifier = EvidenceVerifier(model=settings.gemini_model)
        self.answerer = AnswerGenerator(model=settings.gemini_model)

    def run(self, question: str, progress_callback=None) -> InvestigationRecord:
        """Run the full pipeline for a question, reporting progress via callback."""

        def report(msg: str) -> None:
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)

        record = InvestigationRecord(question=question)

        # Step 1: Plan
        report("Planning investigation...")
        plan = self.planner.plan(question)
        record.plan = plan

        # Step 2: Search + Step 3: Scrape + Step 4: Extract, per sub-question
        all_evidence: List[EvidenceItem] = []
        for i, sub_q in enumerate(plan.sub_questions, start=1):
            report(f"[{i}/{len(plan.sub_questions)}] Searching: {sub_q.question}")
            results: List[SearchResult] = self.searcher.search(sub_q.question)

            if not results:
                report(f"No search results for: {sub_q.question}")
                continue

            for result in results:
                report(f"Reading source: {result.domain or result.url}")
                page: ScrapedPage = self.scraper.scrape(result)
                if not page.success:
                    continue
                evidence = self.extractor.extract(page, sub_q.question)
                all_evidence.extend(evidence)

        record.evidence_items = all_evidence
        report(f"Gathered {len(all_evidence)} evidence item(s) total.")

        # Step 5: Verify
        report("Cross-checking evidence for conflicts...")
        verification = self.verifier.verify(all_evidence)
        record.verification = verification

        # Step 6: Answer
        report("Synthesizing final answer...")
        final_answer = self.answerer.generate(question, all_evidence, verification)
        record.final_answer = final_answer

        report("Investigation complete.")
        return record


# --------------------------------------------------------------------------
# Section 5: Rendering helpers
# --------------------------------------------------------------------------
CONFIDENCE_STAMP_COLORS = {"High": "#2DD4BF", "Medium": "#F2B84B", "Low": "#EF5350"}


def render_investigation(record: InvestigationRecord) -> None:
    """Render a completed InvestigationRecord as rich Streamlit output."""
    final_answer = record.final_answer
    if final_answer is None:
        st.warning("This investigation did not complete successfully.")
        return

    stamp_color = CONFIDENCE_STAMP_COLORS.get(final_answer.confidence_level, "#8B98A5")
    st.markdown(
        f"### Answer "
        f"<span class='confidence-stamp' style='--stamp-color: {stamp_color};'>"
        f"{final_answer.confidence_level} confidence</span>",
        unsafe_allow_html=True,
    )
    st.markdown(final_answer.answer_text)

    if final_answer.confidence_rationale:
        with st.expander("Why this confidence level?"):
            st.write(final_answer.confidence_rationale)

    if final_answer.conflicts_summary:
        with st.expander(f"⚠️ Conflicting information found ({len(final_answer.conflicts_summary)})"):
            for c in final_answer.conflicts_summary:
                st.markdown(f"- {c}")

    if final_answer.sources:
        with st.expander(f"📚 Sources ({len(final_answer.sources)})", expanded=False):
            for src in final_answer.sources:
                st.markdown(f"**[{src.citation_id}]** [{src.title or src.url}]({src.url})  \n*{src.domain}*")
                for claim in src.why_used:
                    st.markdown(f"  - {claim}")

    if record.plan and record.plan.sub_questions:
        with st.expander("🗺️ Investigation plan", expanded=False):
            for sq in record.plan.sub_questions:
                st.markdown(f"- **{sq.question}**  \n  _{sq.rationale}_")


# --------------------------------------------------------------------------
# Section 6: Sidebar
# --------------------------------------------------------------------------
def render_sidebar(settings: Settings) -> None:
    with st.sidebar:
        st.markdown("## 🔎 OSINT AGENT")
        st.caption("Investigative research assistant · Google Gemini")
        st.divider()
        st.markdown(
            f"<span style='color:#8B98A5; font-family:\"IBM Plex Mono\",monospace; "
            f"font-size:0.78rem; text-transform:uppercase; letter-spacing:0.08em;'>Model</span><br>"
            f"<code>{settings.gemini_model}</code>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<br><span style='color:#8B98A5; font-family:\"IBM Plex Mono\",monospace; "
            f"font-size:0.78rem; text-transform:uppercase; letter-spacing:0.08em;'>Max sub-questions</span><br>"
            f"<code>{settings.max_subquestions}</code>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<br><span style='color:#8B98A5; font-family:\"IBM Plex Mono\",monospace; "
            f"font-size:0.78rem; text-transform:uppercase; letter-spacing:0.08em;'>Results per sub-question</span><br>"
            f"<code>{settings.max_search_results_per_subquestion}</code>",
            unsafe_allow_html=True,
        )
        st.divider()
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.history = []
            st.rerun()
        st.divider()
        st.caption(
            "This agent only reports claims explicitly found in web sources, "
            "with citations. It does not verify the underlying truth of those "
            "sources — always review the linked material yourself."
        )


# --------------------------------------------------------------------------
# Section 7: Session state
# --------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history: List[InvestigationRecord] = []


# --------------------------------------------------------------------------
# Section 8: Settings + client init
# --------------------------------------------------------------------------
try:
    settings = load_settings()
    configure_gemini(settings.gemini_api_key)
except Exception as exc:  # noqa: BLE001
    st.error(
        "⚠️ **Configuration Error**\n\n"
        f"{exc}\n\n"
        "Please copy `.env.example` to `.env` and set a valid `GEMINI_API_KEY` "
        "(get one free at https://aistudio.google.com/apikey), then restart the app."
    )
    st.stop()

render_sidebar(settings)
pipeline = InvestigationPipeline(settings=settings)


# --------------------------------------------------------------------------
# Section 9: Main chat UI
# --------------------------------------------------------------------------
st.title("OSINT Investigative Agent")
st.caption(
    "Ask a broad investigative question. The agent will plan sub-questions, "
    "search the web, read sources, extract evidence, cross-check for "
    "conflicts, and return a cited answer."
)

for record in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(record.question)
    with st.chat_message("assistant"):
        render_investigation(record)

question = st.chat_input("Ask an investigative question...")

if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        status_box = st.status("Investigating...", expanded=True)

        def progress_callback(msg: str) -> None:
            status_box.markdown(
                f"<div class='log-line'><span class='log-prompt'>&gt;</span>{msg}</div>",
                unsafe_allow_html=True,
            )

        try:
            record = pipeline.run(question, progress_callback=progress_callback)
            status_box.update(label="Investigation complete", state="complete", expanded=False)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Pipeline crashed for question '{question}': {exc}")
            status_box.update(label="Investigation failed", state="error", expanded=True)
            st.error(
                "Something went wrong while running the investigation. "
                f"Details: {exc}"
            )
            record = InvestigationRecord(question=question)

        render_investigation(record)
        st.session_state.history.append(record)
