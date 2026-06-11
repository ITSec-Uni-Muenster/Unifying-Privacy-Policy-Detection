from __future__ import annotations

import argparse
import json
import re
import ftfy
from pathlib import Path
from typing import Any, Iterable

from bs4 import BeautifulSoup
import trafilatura
from tqdm import tqdm
import os

from openai import OpenAI
from chardet import UniversalDetector

# Konfiguration für KI-Connect 
KI_BASE_URL = "https://chat.kiconnect.nrw/api/v1"
KI_API_KEY = os.getenv("KI_API_KEY")

KI_MODEL_LANGUAGE = "Mistral Small 4 119B"
KI_MODEL_POLICY = "Mistral Small 4 119B"

KI_LANGUAGE_TIMEOUT = 60
KI_LANGUAGE_MAX_CHARS = 4000
KI_POLICY_TIMEOUT = 180
KI_POLICY_MAX_CHARS = 8000


# Wörterbuch mit typischen Mojibake-/Encoding-Fehlern.
# Beispiel: "Ã¼" soll zu "ü" werden.
dict_of_umlaute_errors = {
    'Ã¼': 'ü',
    'Ã¤': 'ä',
    'Ã¶': 'ö',
    'Ã–': 'Ö',
    'ÃŸ': 'ß',
    'Ã ': 'à',
    'Ã¡': 'á',
    'Ã¢': 'â',
    'Ã£': 'ã',
    'Ã¹': 'ù',
    'Ãº': 'ú',
    'Ã»': 'û',
    'Ã™': 'Ù',
    'Ãš': 'Ú',
    'Ã›': 'Û',
    'Ãœ': 'Ü',
    'Ã²': 'ò',
    'Ã³': 'ó',
    'Ã´': 'ô',
    'Ã¨': 'è',
    'Ã©': 'é',
    'Ãª': 'ê',
    'Ã«': 'ë',
    'â‚¬': '€'
}

# Erweitert das Wörterbuch zusätzlich um lowercase-Varianten der Keys,
# damit noch mehr fehlerhafte Zeichenfolgen abgefangen werden.
dict_of_umlaute_errors = {
    **dict_of_umlaute_errors,
    **{key.lower(): value for key, value in dict_of_umlaute_errors.items()}
}


def repair_encoding_errors(text: str) -> str:
    """
    Repariert typische Text-/Encoding-Probleme.

    Schritte:
    1. Allgemeine Unicode-/Mojibake-Reparatur mit ftfy
    2. Zusätzliche manuelle Ersetzungen aus dem alten Repository
    3. Leichte Normalisierung von Leerzeichen und Zeilenumbrüchen

    Rückgabe:
        Bereinigter Text als String.
    """
    if not text:
        return ""

    text = ftfy.fix_text(text)

    for wrong, correct in dict_of_umlaute_errors.items():
        text = text.replace(wrong, correct)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _extract_first_json_object(text: str) -> dict | None:
    """
    Extrahiert robust das erste JSON-Objekt aus einer LLM-Antwort.
    Funktioniert auch, wenn das Modell vor/nach dem JSON zusätzlichen Text erzeugt.
    """
    if not text:
        return None

    text = text.strip()

    # 1. Direkter JSON-Versuch
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 2. JSON-Objekt ab jeder öffnenden Klammer versuchen
    decoder = json.JSONDecoder()

    for i, char in enumerate(text):
        if char != "{":
            continue

        try:
            obj, _ = decoder.raw_decode(text[i:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue

    # 3. Fallback: Label aus freiem Text extrahieren
    label_match = re.search(r'"?label"?\s*:\s*"?(keep|drop)"?', text, flags=re.IGNORECASE)
    reason_match = re.search(r'"?reason"?\s*:\s*"([^"]+)"', text, flags=re.IGNORECASE)

    if label_match:
        return {
            "label": label_match.group(1).lower(),
            "reason": reason_match.group(1).strip() if reason_match else "Reason could not be fully parsed."
        }

    return None
    

def get_ki_client(timeout: int = KI_LANGUAGE_TIMEOUT) -> OpenAI:
    """
    Erstellt den OpenAI-kompatiblen Client für die KI-Connect API.

    Wichtig:
    Der API-Key wird nicht direkt im Code gespeichert.
    Vor dem Start muss im Terminal gesetzt werden:

        export KI_API_KEY='...'
    """
    if not KI_API_KEY:
        raise RuntimeError(
            "KI_API_KEY is not set. "
            "Please run: export KI_API_KEY='your-api-key'"
        )

    return OpenAI(
        api_key=KI_API_KEY,
        base_url=KI_BASE_URL,
        timeout=timeout,
    )


def clean_language_name(language: str | None) -> str:
    """
    Bereinigt nur die Sprachangabe aus der Mistral-Antwort.

    Wichtig:
    Diese Funktion macht kein Mapping.
    Sie wandelt also nicht German -> de um.

    Beispiele:
        "German" bleibt "German"
        "English" bleibt "English"
        "Portuguese" bleibt "Portuguese"
        "mixed" bleibt "mixed"
        "unknown" bleibt "unknown"
    """
    if not language:
        return "unknown"

    language = str(language).strip()

    if not language:
        return "unknown"

    # Nur sehr lange oder kaputte Antworten verhindern
    if len(language) > 50:
        return "unknown"

    return language


def detect_language_with_mistral(text: str) -> tuple[str, float | None]:
    """
    Erkennt die Sprache ausschließlich mit dem KI-Connect Mistral-Modell.

    Kein fastText.
    Kein UI-/Mixed-Precheck.
    Kein Mapping.
    Keine feste Sprachliste.

    Mistral darf die Sprache vollständig ausschreiben:
        German, English, French, Spanish, Arabic, Turkish, Vietnamese, ...

    Rückgabe:
        (language, confidence)
    """
    cleaned = repair_encoding_errors(text).strip()

    if not cleaned:
        return "unknown", 0.0

    client = get_ki_client(KI_LANGUAGE_TIMEOUT)

    sample = cleaned[:KI_LANGUAGE_MAX_CHARS]

    system_prompt = """
You are a strict language detection component in a multilingual privacy-policy processing pipeline.

Your task is to detect the language of the provided text.

You must decide only from the text itself.
Do not use external knowledge.
Do not classify whether the text is a privacy policy.
Do not classify whether the text is a UI menu, cookie banner, footer, navigation, or boilerplate.
Only detect the language of the given text.

Return exactly one valid JSON object.
Do not use markdown.
Do not add explanations outside the JSON.

Language output rule:
- If one language is clearly identifiable, write the full English language name.
- Examples: "German", "English", "French", "Spanish", "Italian", "Dutch", "Portuguese", "Polish", "Turkish", "Arabic", "Russian", "Chinese", "Japanese", "Korean", "Vietnamese", "Greek", "Ukrainian", "Romanian", "Hindi", "Bengali", "Farsi", "Hokkien", "Southern Min", "Sranan Tongo", "Dagbani", "Minangkabau", "Tyap", etc.
- You are not limited to this example list.
- Use the most specific full English language name that best describes the text.

Special values:
- Use "mixed" if the text contains multiple different languages and no single language is clearly dominant.
- Use "unknown" if the language cannot be determined reliably.

Important rules:
1. Treat UI labels, menus, navigation text, footer text, cookie text, and short interface text as normal text for language detection.
2. Return "mixed" if the text is mainly a list of language names such as Deutsch, English, Español, Français, Italiano, Português, Nederlands, Polski, etc.
3. Do not return "mixed" only because the text is a UI menu, footer, cookie banner, or navigation block.
4. Return a normal language if the UI/menu/footer text is clearly written in one language.
5. Return "mixed" only when the text itself contains multiple languages and no single language is clearly dominant.
6. If a text is a language-selection list containing many language names in different languages, return "mixed".
7. If one language clearly dominates, return that language even if a few foreign words, names, buttons, or links appear.
8. Return "unknown" only if the language cannot be identified reliably from the text.
9. The confidence must be a number between 0.0 and 1.0.
10. Return only JSON.

Required JSON format:
{
  "language": "German",
  "confidence": 0.98
}
""".strip()

    try:
        resp = client.chat.completions.create(
            model=KI_MODEL_LANGUAGE,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": sample},
            ],
        )

        raw_answer = resp.choices[0].message.content.strip()
        parsed = _extract_first_json_object(raw_answer)

        if not parsed:
            print("[MISTRAL LANGUAGE RAW RESPONSE]", raw_answer, flush=True)
            return "unknown", 0.0

        language = clean_language_name(parsed.get("language"))

        try:
            confidence = float(parsed.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        confidence = max(0.0, min(1.0, confidence))

        return language, confidence

    except Exception as e:
        print(f"[MISTRAL LANGUAGE DETECTION ERROR] {e}", flush=True)
        return "unknown", 0.0


def detect_language(text: str) -> tuple[str, float | None]:
    """
    Zentrale Sprachdetektion der Pipeline.

    Rückgabe:
        (language, confidence)
    """
    return detect_language_with_mistral(text)


def find_latest_datadir(openwpm_root: Path) -> Path:
    """
    Sucht im OpenWPM-Root nach Ordnern mit dem Muster 'datadir_*'
    und gibt den neuesten (lexikographisch letzten) zurück.

    Nützlich, wenn der Datadir-Ordner nach jedem Crawl neu erzeugt wird.
    """
    candidates = sorted(openwpm_root.glob("datadir_*"))
    if not candidates:
        raise FileNotFoundError(f"Keine datadir_* Ordner gefunden in: {openwpm_root}")
    return candidates[-1]


def iter_input_files(datadir: Path) -> Iterable[Path]:
    """
    Liefert rekursiv alle .ndjson-Dateien aus einem datadir.

    Diese Dateien bilden den Input für die erste Phase der Pipeline.
    """
    yield from datadir.rglob("*.ndjson")


def fallback_bs4_text(html: str) -> str:
    """
    Extrahiert Text aus HTML mit BeautifulSoup als Fallback-Methode.

    Vorgehen:
    1. Entfernt irrelevante HTML-Tags wie script/style/img/iframe
    2. Versucht zuerst typische Hauptcontainer zu finden
    3. Falls nichts Passendes gefunden wird, wird der gesamte Body gelesen

    Rückgabe:
        Extrahierter Klartext.
    """
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "noscript", "svg", "img", "iframe"]):
        tag.decompose()

    preferred_selectors = [
        "main",
        "article",
        '[role="main"]',
        ".privacy-policy",
        "#privacy-policy",
        ".policy",
        "#policy",
        ".content",
        "#content",
        ".main-content",
        "#main-content",
    ]

    for selector in preferred_selectors:
        node = soup.select_one(selector)
        if node:
            text = node.get_text(separator="\n", strip=True)
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = re.sub(r"[ \t]{2,}", " ", text)
            if len(text) > 300:
                return text.strip()

    body = soup.body if soup.body else soup
    text = body.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def extract_text_from_html(html: str, document_type: str) -> str:
    """
    Extrahiert Haupttext aus HTML abhängig vom Dokumenttyp.

    document_type:
    - 'privacy_policy': eher präzise Extraktion
    - 'landing_page': eher recall-orientierte Extraktion

    Zuerst wird trafilatura versucht.
    Falls das fehlschlägt oder keinen Text liefert, wird BeautifulSoup-Fallback genutzt.
    """
    if not html:
        return ""

    if document_type == "privacy_policy":
        options = dict(
            output_format="txt",
            include_comments=False,
            include_tables=False,
            include_images=False,
            favor_precision=True,
            favor_recall=False,
        )
    elif document_type == "landing_page":
        options = dict(
            output_format="txt",
            include_comments=False,
            include_tables=True,
            include_images=False,
            favor_precision=False,
            favor_recall=True,
        )
    else:
        return fallback_bs4_text(html)

    try:
        text = trafilatura.extract(html, **options)
        if text and text.strip():
            return text.strip()
    except Exception:
        pass

    return fallback_bs4_text(html)


def detect_file_encoding(path: Path) -> str:
    """
    Schätzt das Encoding einer Datei mit chardet.

    Rückgabe:
        Erkannter Encoding-Name oder 'utf-8' als Fallback.
    """
    detector = UniversalDetector()
    try:
        with path.open("rb") as f:
            for line in f:
                detector.feed(line)
                if detector.done:
                    break
        detector.close()
        encoding = detector.result.get("encoding")
        return encoding if encoding else "utf-8"
    except Exception:
        return "utf-8"


def read_text_file(path: Path) -> str:
    """
    Liest eine Textdatei robust ein.

    Ablauf:
    1. Encoding automatisch erkennen
    2. Datei mit erkanntem Encoding lesen
    3. Falls das fehlschlägt: UTF-8-Fallback
    4. Falls alles fehlschlägt: leerer String

    Rückgabe:
        Dateiinhalt als String.
    """
    try:
        encoding = detect_file_encoding(path)
        return path.read_text(encoding=encoding, errors="ignore")
    except Exception:
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""


def normalize_to_list(value: Any) -> list[str]:
    """
    Normalisiert einen Wert zu einer Liste von Strings.

    Fälle:
    - None -> []
    - list -> Liste von Strings
    - nichtleerer String -> [string]
    - sonst -> []

    Das ist hilfreich, weil manche JSON-Felder mal String,
    mal Liste, mal leer sein können.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v]
    if isinstance(value, str) and value.strip():
        return [value]
    return []


def process_json_file(path: Path) -> list[dict]:
    """
    Verarbeitet eine .ndjson-Datei und erzeugt extrahierte Text-Records.

    Aus jedem JSON-Record werden, falls vorhanden:
    1. Privacy-Policy-Dateien gelesen und extrahiert
    2. Landing-Page-Datei gelesen und extrahiert

    Rückgabe:
        Liste von Records mit Metadaten und extrahiertem Text.
    """
    records = []

    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except Exception:
                    continue

                domain = data.get("domain", "")
                crawl_id = data.get("crawl_id", "")
                landing_url = data.get("landing_url", "")

                privacy_files = normalize_to_list(data.get("privacy_policy_file"))
                privacy_urls = normalize_to_list(data.get("privacy_policy_url"))

                for j, file_path in enumerate(privacy_files):
                    p = Path(file_path)
                    html = read_text_file(p)
                    if not html:
                        continue

                    text = extract_text_from_html(html, "privacy_policy")
                    if not text:
                        continue

                    url = privacy_urls[j] if j < len(privacy_urls) else ""

                    records.append({
                        "source_ndjson": str(path),
                        "record_index": f"{idx}-pp-{j}",
                        "domain": domain,
                        "crawl_id": crawl_id,
                        "landing_url": landing_url,
                        "document_type": "privacy_policy",
                        "source_file": str(p),
                        "source_url": url,
                        "chars": len(text),
                        "text": text,
                    })

                landing_file = data.get("landing_page")
                if isinstance(landing_file, str) and landing_file.strip():
                    p = Path(landing_file)
                    html = read_text_file(p)
                    if html:
                        text = extract_text_from_html(html, "landing_page")
                        if text:
                            records.append({
                                "source_ndjson": str(path),
                                "record_index": f"{idx}-landing",
                                "domain": domain,
                                "crawl_id": crawl_id,
                                "landing_url": landing_url,
                                "document_type": "landing_page",
                                "source_file": str(p),
                                "source_url": landing_url,
                                "chars": len(text),
                                "text": text,
                            })

    except Exception:
        return []

    return records


def process_input_file(path: Path) -> list[dict]:
    """
    Leitet die Verarbeitung einer Input-Datei an die passende Funktion weiter.

    Aktuell unterstützt:
    - .ndjson -> process_json_file()

    Rückgabe:
        Liste von extrahierten Records.
    """
    if path.suffix.lower() == ".ndjson":
        return process_json_file(path)
    return []


def text_extraction_module(datadir: Path, output: Path) -> None:
    """
    Phase 1 der Pipeline: Text-Extraktion.

    Ablauf:
    1. Sucht alle Input-Dateien im datadir
    2. Verarbeitet jede Datei
    3. Schreibt alle extrahierten Records in eine JSONL-Datei

    Ausgabe:
        JSONL-Datei mit Texten aus Privacy Policies und Landing Pages.
    """
    files = list(iter_input_files(datadir))
    if not files:
        raise FileNotFoundError(f"Keine passenden Input-Dateien gefunden unter: {datadir}")

    output.parent.mkdir(parents=True, exist_ok=True)

    total_records = 0

    with output.open("w", encoding="utf-8") as fout:
        for path in tqdm(files, desc="Extracting"):
            records = process_input_file(path)
            if records:
                for rec in records:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    total_records += 1

    print("Fertig.")
    print(f"Geschriebene Records: {total_records}")
    print(f"Output: {output}")


def language_detection_module(input_jsonl: Path, output_jsonl: Path) -> None:
    """
    Phase 2 der Pipeline: Sprachdetektion.

    Liest die extrahierten Texte aus Phase 1 und verarbeitet nur Records
    vom Typ 'privacy_policy'.

    Pro Record werden ergänzt:
    - bereinigter Text
    - language
    - language_confidence

    Ausgabe:
        JSONL-Datei mit Sprachinformationen.
    """
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    total_in = 0
    total_out = 0
    skipped_non_policy = 0

    with input_jsonl.open("r", encoding="utf-8") as fin, \
         output_jsonl.open("w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc="Language detection"):
            line = line.strip()
            if not line:
                continue

            total_in += 1

            try:
                record = json.loads(line)
            except Exception:
                continue

            if record.get("document_type") != "privacy_policy":
                skipped_non_policy += 1
                continue

            original_text = record.get("text", "")
            cleaned_text = repair_encoding_errors(original_text)
            language, confidence = detect_language(cleaned_text)

            record["text"] = cleaned_text
            record["language"] = language
            record["language_confidence"] = confidence
            record["has_detected_language"] = str(language).strip().lower() not in {"unknown", "mixed"}

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            total_out += 1

    print("Fertig.")
    print(f"Input records gelesen: {total_in}")
    print(f"Nicht-Privacy-Policy übersprungen: {skipped_non_policy}")
    print(f"Output records geschrieben: {total_out}")
    print(f"Output: {output_jsonl}")


def classify_policy_with_mistral(
    text: str,
    language: str = "unknown",
    language_confidence: float | None = None,
) -> tuple[str, str]:
    """
    Klassifiziert einen extrahierten Text als keep/drop.

    Wichtig:
    - Die Sprache wurde vorher in Step 2 erkannt.
    - Die Sprache dient hier nur als Kontext.
    - Die inhaltliche Entscheidung trifft der Classifier.
    - mixed/unknown wird NICHT automatisch gedroppt.

    Rückgabe:
        (label, reason)

    label:
        "keep" oder "drop"

    reason:
        kurze faktische Begründung
    """
    cleaned = repair_encoding_errors(text).strip()

    if not cleaned:
        return "drop", "The text is empty."

    client = get_ki_client(KI_POLICY_TIMEOUT)

    sample = cleaned[:KI_POLICY_MAX_CHARS]

    system_prompt = """
You are a text-classification assistant for privacy-policy crawling.

Decide whether the provided text content should be kept as privacy-relevant content or dropped.

Important scope:
- Classify only the provided text content.
- Do not classify based on the URL.
- Do not classify based on the domain.
- Do not classify based on the detected language.
- The detected language is only context.

The text may:
- be in any language
- be noisy, incomplete, or badly extracted
- contain HTML, boilerplate, cookie banners, navigation, duplicated blocks
- contain fragments only
- include mixed privacy + unrelated content

Goal:
Keep real privacy-relevant content, but drop obvious false positives.

Return exactly one JSON object:
{"label":"keep|drop","reason":"one short sentence"}

Return "keep" if the text clearly contains concrete privacy-relevant information, such as:
- privacy policy, privacy notice, data protection statement
- personal data collection, processing, storage, sharing, deletion
- legal bases such as consent, legitimate interests, GDPR, DSGVO, CCPA
- user/data-subject rights
- controller, processor, DPO, supervisory authority
- retention periods, data categories, recipients, international transfers
- privacy contact information
- cookie or consent explanations as part of broader privacy/data processing
- terms of service with clear personal-data processing information
- privacy contact information such as DPO, privacy office, data protection authority, or data-subject rights contact

Return "drop" if the text is mainly:
- only a cookie banner or consent UI with buttons like accept/reject/manage
- only language selection or a list of language names
- only UI menu, navigation, footer, header, login, FAQ, contact, about page
- terms without clear privacy relevance
- imprint, accessibility, shipping, refund, unrelated boilerplate
- raw code, empty, near-empty, or unrelated content

Very important negative rules:
- A language-selection list must be classified as "drop".
- A list of language names such as Deutsch, English, Español, Français, Italiano, Nederlands, Polski, Português, etc. must be classified as "drop".
- Do not classify a language menu as "keep" only because it comes from a consent or privacy-related URL.
- Do not classify a text as "keep" only because it contains words like privacy, consent, cookies, terms, GDPR, or policy in a menu or link.
- Keep consent-related text only if it explains personal-data processing, legal basis, tracking, cookies, identifiers, user rights, or similar privacy-relevant facts.
- If cookies are only button-based UI without explanation, return "drop".
- If ambiguous and no clear privacy evidence is present in the text, return "drop".
- A mere link or reference to a privacy policy is not enough for "keep".
- Contact information alone is not enough for "keep" unless it is clearly privacy-specific contact information such as DPO, privacy office, data protection authority, or data-subject rights contact.

Reason:
- one short factual sentence only
- use concrete evidence from the text
- do not speculate or add information not present in the text
- keep it short and factual

Return only JSON.
Do not use markdown.
Do not add explanations outside the JSON.
""".strip()

    user_prompt = f"""
Detected language: {language}
Language confidence: {language_confidence}

Text:
\"\"\"
{sample}
\"\"\"
""".strip()

    try:
        resp = client.chat.completions.create(
            model=KI_MODEL_POLICY,
            temperature=0.0,
            top_p=1.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        raw_answer = resp.choices[0].message.content.strip()

        parsed = _extract_first_json_object(raw_answer)

        if not parsed:
            print("[MISTRAL POLICY RAW RESPONSE]", raw_answer, flush=True)
            return "drop", "Mistral did not return valid JSON."

        label = str(parsed.get("label", "drop")).strip().lower()
        reason = str(parsed.get("reason", "")).strip()

        if label not in {"keep", "drop"}:
            print("[MISTRAL POLICY INVALID LABEL]", raw_answer, flush=True)
            return "drop", "Mistral returned an invalid label."

        if not reason:
            reason = "Mistral returned no reason."

        return label, reason

    except Exception as e:
        print(f"[MISTRAL POLICY DETECTION ERROR] {e}", flush=True)
        return "drop", "Request to Mistral failed."


def policy_detection_module(input_jsonl: Path, output_jsonl: Path) -> None:
    """
    Phase 3 der Pipeline: Policy Detection / Keep-Drop-Classification.

    Input:
        extraction_lang.jsonl aus Step 2

    Wichtig:
        Die Language Detection entscheidet nur die Sprache.
        Die Classification entscheidet keep/drop.

        Deshalb wird hier NICHT automatisch nach language, mixed,
        unknown oder has_detected_language gefiltert.

    Pro Record werden ergänzt:
        - policy_detection_label
        - policy_detection_reason
    """
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    total_in = 0
    total_out = 0
    kept = 0
    dropped = 0

    with input_jsonl.open("r", encoding="utf-8") as fin, \
         output_jsonl.open("w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc="Policy detection"):
            line = line.strip()
            if not line:
                continue

            total_in += 1

            try:
                record = json.loads(line)
            except Exception:
                continue

            text = record.get("text", "")
            language = record.get("language", "unknown")
            language_confidence = record.get("language_confidence")

            label, reason = classify_policy_with_mistral(
                text=text,
                language=language,
                language_confidence=language_confidence,
            )

            out_record = {
                "record_index": record.get("record_index"),
                "domain": record.get("domain"),
                "crawl_id": record.get("crawl_id"),
                "source_url": record.get("source_url"),
                "source_file": record.get("source_file"),
                "document_type": record.get("document_type"),

                "language": language,
                "language_confidence": language_confidence,
                "has_detected_language": record.get("has_detected_language"),

                "chars": record.get("chars"),

                "policy_detection_label": label,
                "policy_detection_reason": reason,

                "text_preview": repair_encoding_errors(text)[:500],
            }

            fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            total_out += 1

            if label == "keep":
                kept += 1
            else:
                dropped += 1

    print("Fertig.")
    print(f"Input records gelesen: {total_in}")
    print(f"Output records geschrieben: {total_out}")
    print(f"Keep: {kept}")
    print(f"Drop: {dropped}")
    print(f"Output: {output_jsonl}")


def main():
    """
    Einstiegspunkt des Skripts.

    Aufgaben:
    1. Liest Kommandozeilenargumente
    2. Führt Phase 1: Text-Extraktion aus
    3. Führt Phase 2: Sprachdetektion aus
    4. Führt Phase 3: Klassification aus

    Damit wird die bisherige 3-stufige Pipeline vollständig gestartet.
    """
    parser = argparse.ArgumentParser(description="Privacy Policy Toolchain")

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Pfad zu datadir_*"
    )

    parser.add_argument(
        "--extraction-output",
        type=Path,
        default=Path("../results/extraction.jsonl"),
        help="Output Datei für Text-Extraktion"
    )

    parser.add_argument(
        "--language-output",
        type=Path,
        default=Path("../results/extraction_lang.jsonl"),
        help="Output Datei für Language Detection"
    )
    
    parser.add_argument(
        "--policy-output",
        type=Path,
        default=Path("../results/policy_detection.jsonl"),
        help="Output Datei für Policy Detection"
    )

    args = parser.parse_args()

    datadir = find_latest_datadir(args.input)

    print("\n==============================")
    print("STEP 1: TEXT EXTRACTION")
    print("==============================")
    print(f"Arbeite auf: {datadir}")
    text_extraction_module(datadir, args.extraction_output)

    print("\n==============================")
    print("STEP 2: LANGUAGE DETECTION")
    print("==============================")
    language_detection_module(args.extraction_output,args.language_output)

    print("\n==============================")
    print("STEP 3: POLICY DETECTION")
    print("==============================")
    policy_detection_module(args.language_output, args.policy_output)


    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()