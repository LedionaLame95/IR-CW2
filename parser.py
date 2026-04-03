"""
parse_clefip.py  —  Standalone CLEF-IP patent XML parser
=========================================================

Parses patent XML files from data/raw/clef_ip/ and saves a structured
JSONL dataframe to data/interim/parsed_patents.jsonl.

Improvements over the Step 1 notebook parser:
  1. Extracts the UCID from the root element attribute (not just doc-number)
  2. Filters for English-only text in title, abstract, claims, description
  3. Handles multilingual patents gracefully (falls back to any language if
     English is missing)
  4. Runs as a standalone script, decoupled from the notebook
  5. Produces per-field output ready for BM25F (Step 2)

Usage:
    python parse_clefip.py                          # parse first 2000 files
    python parse_clefip.py --subset 5000            # parse first 5000 files
    python parse_clefip.py --all                    # parse entire corpus
    python parse_clefip.py --sample 1000 --seed 42  # random sample of 1000
"""

from pathlib import Path
import argparse
import json
import random
import re
import sys
import time

import pandas as pd
from lxml import etree
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Paths  (adjust PROJECT_ROOT if you run from a different working directory)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR      = PROJECT_ROOT / "data" / "raw" / "clef_ip"
INTERIM_DIR  = PROJECT_ROOT / "data" / "interim"
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

# output file names
OUTPUT_JSONL   = INTERIM_DIR / "parsed_patents.jsonl"
FAILURES_JSON  = INTERIM_DIR / "parse_failures.json"
MANIFEST_TXT   = INTERIM_DIR / "parsed_subset_files.txt"


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------
_MULTI_SPACE = re.compile(r"\s+")


def _clean(text: str) -> str:
    """Collapse whitespace, strip leading/trailing blanks."""
    if not text:
        return ""
    return _MULTI_SPACE.sub(" ", text).strip()


def _gather_text(elements) -> str:
    """Recursively collect all inner text from a list of lxml elements."""
    parts = []
    for el in elements:
        # itertext() yields all text in el and its descendants
        parts.append(" ".join(el.itertext()))
    return _clean(" ".join(parts))


# ---------------------------------------------------------------------------
# Language-aware field extraction
# ---------------------------------------------------------------------------
def _get_field_en(root, tag_local_name: str) -> str:
    """
    Extract text from a field, preferring English (lang='en' / 'EN').

    CLEF-IP XML convention:
        <abstract lang="EN"> ... </abstract>
        <abstract lang="FR"> ... </abstract>

    The @lang attribute may also appear as @xml:lang in some files.
    Strategy:
        1. Try elements whose @lang is 'en' or 'EN'
        2. Try elements whose @xml:lang is 'en' or 'EN'
        3. Fall back to first element with no @lang (assumed English)
        4. Fall back to any element (multilingual mix)
    """
    # find all elements with matching local-name, ignoring namespaces
    all_els = root.xpath(f'//*[local-name()="{tag_local_name}"]')

    if not all_els:
        return ""

    # 1) explicit @lang = en/EN
    en_els = [e for e in all_els if (e.get("lang") or "").lower() == "en"]
    if en_els:
        return _gather_text(en_els)

    # 2) @xml:lang = en/EN
    en_els = [e for e in all_els
              if (e.get("{http://www.w3.org/XML/1998/namespace}lang") or "").lower() == "en"]
    if en_els:
        return _gather_text(en_els)

    # 3) elements with no lang attribute at all (common for single-language files)
    no_lang = [e for e in all_els if e.get("lang") is None
               and e.get("{http://www.w3.org/XML/1998/namespace}lang") is None]
    if no_lang:
        return _gather_text(no_lang)

    # 4) last resort: take everything (better than empty)
    return _gather_text(all_els)


def _get_title_en(root) -> str:
    """
    Extract English invention title.
    Tries <invention-title> first, then <title>.
    """
    title = _get_field_en(root, "invention-title")
    if not title:
        title = _get_field_en(root, "title")
    return title


def _get_ucid(root, xml_path: Path) -> str:
    """
    Extract the UCID (Unique Document Identifier).

    Priority:
        1. @ucid attribute on the root <patent-document> element
        2. @ucid on any element (some files nest it)
        3. <doc-number> inside <publication-reference>
        4. Filename stem as fallback
    """
    # 1) root attribute
    ucid = root.get("ucid", "").strip()
    if ucid:
        return ucid

    # 2) any element with @ucid
    ucid_els = root.xpath('//*[@ucid]')
    if ucid_els:
        return ucid_els[0].get("ucid", "").strip()

    # 3) doc-number inside publication-reference
    doc_nums = root.xpath(
        '//*[local-name()="publication-reference"]//*[local-name()="doc-number"]'
    )
    if doc_nums and doc_nums[0].text:
        return doc_nums[0].text.strip()

    # 4) fallback to filename
    return xml_path.stem


def _get_ipc(root) -> str:
    """Extract IPC classification codes."""
    ipc_els = root.xpath('//*[local-name()="classification-ipc"]')
    if ipc_els:
        return _gather_text(ipc_els)

    # alternate tag name used in some CLEF-IP files
    ipc_els = root.xpath('//*[local-name()="classification-ipcr"]')
    if ipc_els:
        return _gather_text(ipc_els)

    return ""


# ---------------------------------------------------------------------------
# Main single-file parser
# ---------------------------------------------------------------------------
def parse_patent_xml(xml_path: Path) -> dict:
    """
    Parse one CLEF-IP patent XML file.

    Returns a dict with:
        ucid, title, abstract, claims, description, ipc, source_file

    Raises ValueError if no usable text is found.
    """
    parser = etree.XMLParser(recover=True, huge_tree=True)
    tree = etree.parse(str(xml_path), parser)
    root = tree.getroot()

    ucid        = _get_ucid(root, xml_path)
    title       = _get_title_en(root)
    abstract    = _get_field_en(root, "abstract")
    claims      = _get_field_en(root, "claims")
    description = _get_field_en(root, "description")
    ipc         = _get_ipc(root)

    if not any([title, abstract, claims, description]):
        raise ValueError(f"No usable English text in {xml_path.name}")

    return {
        "ucid":        ucid,
        "title":       title,
        "abstract":    abstract,
        "claims":      claims,
        "description": description,
        "ipc":         ipc,
        "source_file": str(xml_path),
    }


# ---------------------------------------------------------------------------
# Batch parsing
# ---------------------------------------------------------------------------
def discover_xml_files(raw_dir: Path) -> list[Path]:
    """Recursively find all .xml files under raw_dir, sorted."""
    files = sorted(raw_dir.rglob("*.xml"))
    if not files:
        raise FileNotFoundError(
            f"No XML files found under {raw_dir}.\n"
            f"Place CLEF-IP XMLs in {raw_dir}/"
        )
    return files


def select_subset(all_files: list[Path], subset_size: int | None,
                  use_sample: bool, seed: int) -> list[Path]:
    """Pick a deterministic subset of files."""
    if subset_size is None or subset_size >= len(all_files):
        return all_files
    if use_sample:
        rng = random.Random(seed)
        return rng.sample(all_files, subset_size)
    return all_files[:subset_size]


def parse_corpus(files: list[Path]) -> tuple[pd.DataFrame, list[dict]]:
    """
    Parse a list of XML files into a DataFrame + failure log.

    Returns (docs_df, failures)
    """
    records  = []
    failures = []

    for xml_path in tqdm(files, desc="Parsing patents"):
        try:
            rec = parse_patent_xml(xml_path)
            records.append(rec)
        except Exception as exc:
            failures.append({"file": str(xml_path), "error": repr(exc)})

    if not records:
        raise ValueError("Parsing produced zero usable patent records.")

    docs_df = pd.DataFrame(records)
    docs_df = docs_df.drop_duplicates(subset=["ucid"]).reset_index(drop=True)
    return docs_df, failures


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------
def save_outputs(docs_df: pd.DataFrame, failures: list[dict],
                 files_used: list[Path]):
    """Write JSONL, failure log, and file manifest to INTERIM_DIR."""

    docs_df.to_json(OUTPUT_JSONL, orient="records", lines=True, force_ascii=False)
    print(f"  Parsed patents  → {OUTPUT_JSONL}  ({len(docs_df):,} records)")

    FAILURES_JSON.write_text(json.dumps(failures, indent=2), encoding="utf-8")
    print(f"  Failure log     → {FAILURES_JSON}  ({len(failures):,} failures)")

    MANIFEST_TXT.write_text("\n".join(str(p) for p in files_used), encoding="utf-8")
    print(f"  File manifest   → {MANIFEST_TXT}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Parse CLEF-IP patent XMLs into a structured JSONL."
    )
    ap.add_argument("--subset", type=int, default=2000,
                    help="Number of files to parse (default: 2000)")
    ap.add_argument("--all", action="store_true",
                    help="Parse the entire corpus (ignores --subset)")
    ap.add_argument("--sample", type=int, default=None,
                    help="Use a random sample of N files instead of first N")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for --sample (default: 42)")
    ap.add_argument("--raw-dir", type=str, default=None,
                    help=f"Path to raw XML directory (default: {RAW_DIR})")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir) if args.raw_dir else RAW_DIR

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Raw XML dir  : {raw_dir}")
    print()

    # discover files
    all_files = discover_xml_files(raw_dir)
    print(f"Total XML files found: {len(all_files):,}")

    # select subset
    subset_size = None if args.all else (args.sample or args.subset)
    use_sample  = args.sample is not None
    files = select_subset(all_files, subset_size, use_sample, args.seed)
    print(f"Files to parse: {len(files):,}")
    print()

    # parse
    t0 = time.time()
    docs_df, failures = parse_corpus(files)
    elapsed = time.time() - t0

    print()
    print(f"Parsing complete in {elapsed:.1f}s")
    print(f"  Usable patents : {len(docs_df):,}")
    print(f"  Failures       : {len(failures):,}")

    # field coverage summary
    print()
    print("Field coverage:")
    for field in ["title", "abstract", "claims", "description"]:
        n = (docs_df[field].str.strip() != "").sum()
        pct = 100 * n / len(docs_df)
        avg_words = docs_df[field].str.split().str.len().mean()
        print(f"  {field:>12}: {n:>6,} / {len(docs_df):,}  ({pct:5.1f}%)  "
              f"avg {avg_words:.0f} words")

    # save
    print()
    save_outputs(docs_df, failures, files)
    print("\nDone. Load the output in your notebook with:")
    print(f'  docs_df = pd.read_json("{OUTPUT_JSONL}", lines=True)')


if __name__ == "__main__":
    main()