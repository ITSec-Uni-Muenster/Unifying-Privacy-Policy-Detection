from __future__ import annotations

import json
import os
import signal
import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st

from ppt import (
    find_latest_datadir,
    text_extraction_module,
    language_detection_module,
    policy_detection_module,
)


# Streamlit page configuration

st.set_page_config(
    page_title="Unifying-Privacy-Policy-Detection GUI",
    layout="wide",
)


# Button styling

st.markdown(
    """
    <style>
    /* Primary buttons: green */
    button[kind="primary"] {
        background-color: #2e7d32 !important;
        color: white !important;
        border: 1px solid #2e7d32 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }

    button[kind="primary"]:hover {
        background-color: #256628 !important;
        border: 1px solid #256628 !important;
        color: white !important;
    }

    /* Secondary buttons: grey */
    button[kind="secondary"] {
        background-color: #d9d9d9 !important;
        color: black !important;
        border: 1px solid #a6a6a6 !important;
        border-radius: 8px !important;
    }

    button[kind="secondary"]:hover {
        background-color: #c9c9c9 !important;
        border: 1px solid #8c8c8c !important;
        color: black !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Helper functions

def init_session_state() -> None:
    if "show_tranco_lists" not in st.session_state:
        st.session_state.show_tranco_lists = False

    if "last_message" not in st.session_state:
        st.session_state.last_message = ""

    if "last_error" not in st.session_state:
        st.session_state.last_error = ""

    if "crawler_process" not in st.session_state:
        st.session_state.crawler_process = None

    if "crawler_pid" not in st.session_state:
        st.session_state.crawler_pid = None

    if "crawler_log_file" not in st.session_state:
        st.session_state.crawler_log_file = None

    if "crawler_running" not in st.session_state:
        st.session_state.crawler_running = False

    if "show_results" not in st.session_state:
        st.session_state.show_results = False


def clear_status() -> None:
    st.session_state.last_message = ""
    st.session_state.last_error = ""


def hide_results() -> None:
    st.session_state.show_results = False


def show_status() -> None:
    if st.session_state.last_message:
        st.success(st.session_state.last_message)

    if st.session_state.last_error:
        st.error(st.session_state.last_error)


def get_paths() -> dict[str, Path]:
    """
    Central path configuration.

    Assumption:
        gui_app.py is located in:
        privacy_policy_toolchain/code/gui_app.py

    OpenWPM root:
        ~/OpenWPM

    GUI-safe crawler script:
        ~/OpenWPM/run_crawler_gui.sh

    Results folder:
        privacy_policy_toolchain/results/
    """
    code_dir = Path(__file__).resolve().parent
    toolchain_root = code_dir.parent
    results_dir = toolchain_root / "results"
    logs_dir = toolchain_root / "logs"

    openwpm_root = Path.home() / "OpenWPM"
    crawler_script = openwpm_root / "run_crawler_gui.sh"

    extraction_output = results_dir / "extraction.jsonl"
    language_output = results_dir / "extraction_lang.jsonl"
    policy_output = results_dir / "policy_detection.jsonl"

    return {
        "code_dir": code_dir,
        "toolchain_root": toolchain_root,
        "results_dir": results_dir,
        "logs_dir": logs_dir,
        "openwpm_root": openwpm_root,
        "crawler_script": crawler_script,
        "extraction_output": extraction_output,
        "language_output": language_output,
        "policy_output": policy_output,
        "tranco_dir": Path("/home/openwpm/tranco"),
    }


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []

    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception:
                continue

            if isinstance(obj, dict):
                rows.append(obj)

    return rows


def safe_latest_datadir(openwpm_root: Path) -> Path:
    if not openwpm_root.exists():
        raise FileNotFoundError(f"OpenWPM root not found: {openwpm_root}")

    return find_latest_datadir(openwpm_root)

def get_tranco_lists(tranco_dir: Path) -> list[Path]:
    """
    Findet alle Tranco-Listen (.csv) im angegebenen Ordner.
    """
    if not tranco_dir.exists():
        return []

    return sorted(tranco_dir.glob("*.csv"))


def is_crawler_running() -> bool:
    process = st.session_state.get("crawler_process")

    if process is None:
        st.session_state.crawler_running = False
        return False

    return_code = process.poll()

    if return_code is None:
        st.session_state.crawler_running = True
        return True

    st.session_state.crawler_running = False

    if return_code == 0:
        st.session_state.last_message = "Crawler completed successfully."
    else:
        st.session_state.last_error = f"Crawler stopped or failed with return code: {return_code}"

    return False


def start_crawler_script(
    crawler_script: Path,
    logs_dir: Path,
    tranco_file: Path,
) -> tuple[bool, str]:
    """
    Starts the crawler in the background.

    Important:
    Popen is used instead of subprocess.run, so the GUI remains usable
    and the crawler can be aborted from the interface.
    """
    if not crawler_script.exists():
        return False, f"Crawler script not found: {crawler_script}"

    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "crawler_gui.log"

    try:
        log_handle = log_file.open("w", encoding="utf-8", errors="ignore")

        process = subprocess.Popen(
            ["bash", str(crawler_script), str(tranco_file)],
            cwd=str(crawler_script.parent),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )

        log_handle.close()

        st.session_state.crawler_process = process
        st.session_state.crawler_pid = process.pid
        st.session_state.crawler_log_file = str(log_file)
        st.session_state.crawler_running = True

        return True, f"Crawler started. PID: {process.pid}"

    except Exception as e:
        return False, str(e)


def abort_crawler() -> tuple[bool, str]:
    """
    Sends a CTRL+C-like signal to the crawler process group.
    Already crawled pages remain in the current datadir_* folder.
    """
    process = st.session_state.get("crawler_process")

    if process is None:
        st.session_state.crawler_running = False
        return False, "No crawler process is running."

    if process.poll() is not None:
        st.session_state.crawler_running = False
        return False, "Crawler process is already stopped."

    try:
        pid = process.pid
        os.killpg(pid, signal.SIGINT)

        st.session_state.crawler_running = False

        return True, "Crawler abort signal sent successfully."

    except Exception as e:
        return False, str(e)


def handle_abort_query_param() -> None:
    """
    Handles the red Abort Crawler button.

    The HTML button sets:
        ?abort_crawler=1

    Then this function sends a CTRL+C-like signal to the crawler.
    """
    if st.query_params.get("abort_crawler") == "1":
        clear_status()

        ok, message = abort_crawler()

        if ok:
            st.session_state.last_message = message
        else:
            st.session_state.last_error = message

        st.query_params.clear()
        st.rerun()


def show_results(
    extraction_output: Path,
    language_output: Path,
    policy_output: Path,
) -> None:
    """
    Displays selectable result views.

    Extraction Results:
        Shows only extracted privacy-policy pages.

    Language Detection Results:
        Shows language and confidence.

    Policy Detection Results:
        Shows keep/drop and reason.

    Full Results:
        Shows Domain, Language, Classification, Reason.
    """

    st.markdown("**Select result view**")

    result_view = st.radio(
        label="",
        options=[
            "Extraction Results",
            "Language Detection Results",
            "Policy Detection Results",
            "Full Results",
        ],
        horizontal=True,
        label_visibility="collapsed",
    )

    # 1. Extraction Results

    if result_view == "Extraction Results":
        rows = load_jsonl(extraction_output)

        if not rows:
            st.info("No extraction results found. Please run Text Extraction first.")
            return

        display_rows = []

        for row in rows:
            if row.get("document_type") != "privacy_policy":
                continue

            display_rows.append(
                {
                    "Domain": row.get("domain", ""),
                    "Source URL": row.get("source_url", ""),
                    "Chars": row.get("chars", ""),
                }
            )

        if not display_rows:
            st.info("No privacy-policy pages found in extraction results.")
            return

        df = pd.DataFrame(display_rows)

        st.dataframe(
            df,
            width="stretch",
            hide_index=True,
        )

    # 2. Language Detection Results
        
    elif result_view == "Language Detection Results":
        rows = load_jsonl(language_output)

        if not rows:
            st.info("No language detection results found. Please run Language Detection first.")
            return

        display_rows = []

        for row in rows:
            display_rows.append(
                {
                    "Domain": row.get("domain", ""),
                    "Language": row.get("language", ""),
                    "Confidence": row.get("language_confidence", ""),
                }
            )

        df = pd.DataFrame(display_rows)

        st.dataframe(
            df,
            width="stretch",
            hide_index=True,
        )

    # 3. Policy Detection Results
    
    elif result_view == "Policy Detection Results":
        rows = load_jsonl(policy_output)

        if not rows:
            st.info("No policy detection results found. Please run Policy Classification first.")
            return

        display_rows = []

        for row in rows:
            display_rows.append(
                {
                    "Domain": row.get("domain", ""),
                    "Classification": row.get("policy_detection_label", ""),
                    "Reason": row.get("policy_detection_reason", ""),
                }
            )

        df = pd.DataFrame(display_rows)

        st.dataframe(
            df,
            width="stretch",
            hide_index=True,
        )

    # 4. Full Results
    
    elif result_view == "Full Results":
        rows = load_jsonl(policy_output)

        if not rows:
            st.info("No full results found. Please run Policy Classification first.")
            return

        display_rows = []

        for row in rows:
            display_rows.append(
                {
                    "Domain": row.get("domain", ""),
                    "Language": row.get("language", ""),
                    "Classification": row.get("policy_detection_label", ""),
                    "Reason": row.get("policy_detection_reason", ""),
                }
            )

        df = pd.DataFrame(display_rows)

        st.dataframe(
            df,
            width="stretch",
            hide_index=True,
        )

# Main GUI

init_session_state()
paths = get_paths()
handle_abort_query_param()

st.title("Unifying-Privacy-Policy-Detection GUI")

st.write(
    "This project crawls websites with OpenWPM, searches for potential privacy-policy "
    "pages, extracts their text, detects the language of the extracted policy texts, "
    "and classifies whether each text should be kept as privacy-relevant content or "
    "dropped as a false positive. "
    "The GUI allows the user to start the crawler, run the toolchain steps separately "
    "or as a full toolchain, and inspect the generated result files without using "
    "command-line commands directly."
)

show_status()

st.divider()


# 1. Crawler

st.header("1. Crawler")

st.write(
    "The crawler starts the OpenWPM-based crawling process through "
    "`run_crawler_gui.sh`. It creates or updates a `datadir_*` folder. "
    "The crawler can also be aborted; already crawled pages remain stored."
)

openwpm_root = paths["openwpm_root"]
crawler_script = paths["crawler_script"]

if not openwpm_root.exists():
    st.warning(f"OpenWPM root not found: {openwpm_root}")

if not crawler_script.exists():
    st.warning(f"Crawler script not found: {crawler_script}")

try:
    latest_datadir = safe_latest_datadir(openwpm_root)
    st.caption(f"Latest datadir automatically used by the toolchain: {latest_datadir}")
except Exception as e:
    latest_datadir = None
    st.caption(f"No latest datadir found yet: {e}")

crawler_running = is_crawler_running()

tranco_lists = get_tranco_lists(paths["tranco_dir"])

selected_tranco = None

if "selected_tranco" in st.session_state:
    selected_tranco = st.session_state.selected_tranco

crawler_col0, crawler_col1, crawler_col2, crawler_col3 = st.columns([1, 1, 1, 1.3])

with crawler_col0:

    if st.button("Select List"):
        st.session_state.show_tranco_lists = True

    if selected_tranco is not None:
        st.caption(f"Selected list: {selected_tranco.name}")


with crawler_col1:
    start_disabled = (
        crawler_running
        or selected_tranco is None
    )

    if st.button("Start Crawler", type="primary", disabled=start_disabled):
        clear_status()
        hide_results()

        ok, message = start_crawler_script(
            crawler_script=crawler_script,
            logs_dir=paths["logs_dir"],
            tranco_file=selected_tranco,
        )

        if ok:
            st.session_state.last_message = message
        else:
            st.session_state.last_error = message

        st.rerun()


with crawler_col2:
    if crawler_running:
        st.markdown(
            """
            <a href="?abort_crawler=1" target="_self"
               style="
                   display: inline-block;
                   background-color: #c62828;
                   color: white;
                   padding: 0.45rem 0.85rem;
                   border-radius: 8px;
                   text-decoration: none;
                   font-weight: 600;
                   border: 1px solid #c62828;
                   text-align: center;
                   font-size: 0.9rem;
                   line-height: 1.4;
               ">
               Abort Crawler
            </a>
            """,
            unsafe_allow_html=True,
        )


with crawler_col3:
    if crawler_running:
        st.caption("Crawler is running.")
    else:
        st.caption("Crawler is not running.")


# Show Tranco lists AFTER clicking "Select List"

if st.session_state.show_tranco_lists:

    if not tranco_lists:
        st.warning(f"No Tranco lists found in: {paths['tranco_dir']}")

    else:
        tranco_names = [p.name for p in tranco_lists]

        selected_tranco_name = st.radio(
            label="",
            options=tranco_names,
            horizontal=True,
            label_visibility="collapsed",
            key="tranco_radio",
            index=None,
        )

        # Only after the user REALLY selected something
        if selected_tranco_name is not None:

            selected_tranco = next(
                p for p in tranco_lists
                if p.name == selected_tranco_name
            )

            # Save selected list
            st.session_state.selected_tranco = selected_tranco

            # Hide list again
            st.session_state.show_tranco_lists = False

            st.rerun()

st.divider()


# 2. Toolchain

st.header("2. Toolchain")

st.write(
    "The toolchain can be executed step by step or completely. "
    "Text Extraction automatically uses the latest `datadir_*` folder. "
    "Run Full Toolchain executes Text Extraction, Language Detection and "
    "Policy Classification, but it does not start the crawler."
)

st.caption("Individual pipeline steps are on the left. The full toolchain run is separated on the right.")

col1, col2, col3, separator_col, col4 = st.columns([1, 1, 1, 0.08, 1])

with col1:
    if st.button("Run Text Extraction"):
        clear_status()
        hide_results()

        try:
            datadir = safe_latest_datadir(paths["openwpm_root"])

            with st.spinner(f"Running Text Extraction with: {datadir}"):
                text_extraction_module(
                    datadir=datadir,
                    output=paths["extraction_output"],
                )

            st.session_state.last_message = "Text Extraction completed successfully."

        except Exception as e:
            st.session_state.last_error = f"Text Extraction failed: {e}"

with col2:
    if st.button("Run Language Detection"):
        clear_status()
        hide_results()

        try:
            if not paths["extraction_output"].exists():
                raise FileNotFoundError(
                    f"Extraction output not found: {paths['extraction_output']}"
                )

            with st.spinner("Running Language Detection..."):
                language_detection_module(
                    input_jsonl=paths["extraction_output"],
                    output_jsonl=paths["language_output"],
                )

            st.session_state.last_message = "Language Detection completed successfully."

        except Exception as e:
            st.session_state.last_error = f"Language Detection failed: {e}"

with col3:
    if st.button("Run Policy Classification"):
        clear_status()
        hide_results()

        try:
            if not paths["language_output"].exists():
                raise FileNotFoundError(
                    f"Language output not found: {paths['language_output']}"
                )

            with st.spinner("Running Policy Classification..."):
                policy_detection_module(
                    input_jsonl=paths["language_output"],
                    output_jsonl=paths["policy_output"],
                )

            st.session_state.last_message = "Policy Classification completed successfully."

        except Exception as e:
            st.session_state.last_error = f"Policy Classification failed: {e}"


with separator_col:
    st.markdown(
        """
        <div style="
            border-left: 2px solid #999999;
            height: 38px;
            margin-top: 2px;
        "></div>
        """,
        unsafe_allow_html=True,
    )


with col4:
    if st.button("Run Full Toolchain", type="primary"):
        clear_status()
        hide_results()

        try:
            datadir = safe_latest_datadir(paths["openwpm_root"])

            with st.spinner(f"Running Full Toolchain with: {datadir}"):
                text_extraction_module(
                    datadir=datadir,
                    output=paths["extraction_output"],
                )

                language_detection_module(
                    input_jsonl=paths["extraction_output"],
                    output_jsonl=paths["language_output"],
                )

                policy_detection_module(
                    input_jsonl=paths["language_output"],
                    output_jsonl=paths["policy_output"],
                )

            st.session_state.last_message = "Full Toolchain completed successfully."

        except Exception as e:
            st.session_state.last_error = f"Full Toolchain failed: {e}"

st.divider()


# 3. Results

st.header("3. Results")

st.write(
    "Click the button below to display selectable result views for the different "
    "pipeline output files."
)

if st.button("Show Results", type="primary"):
    st.session_state.show_results = True

if st.session_state.show_results:
    show_results(
        extraction_output=paths["extraction_output"],
        language_output=paths["language_output"],
        policy_output=paths["policy_output"],
    )