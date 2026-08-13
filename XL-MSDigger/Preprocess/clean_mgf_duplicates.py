#!/usr/bin/env python3
"""
Study-specific MGF duplicate cleaner for Waters/ProteoWizard-derived DDA XL-MS data.

Rules for records sharing the same SCANS value:
1) If MS2 peak lists differ -> ABORT that file (manual review required).
2) If MS2 peak lists are identical:
   a) If precursor charges differ -> keep the highest charge.
   b) If the retained highest-charge candidates still have the same charge:
      - if precursor m/z values differ by ~integer isotope spacing
        (n * 1.00335483507 / z), keep the lowest precursor m/z;
      - otherwise keep the first record in file order.

The chosen MGF spectrum block is preserved byte-for-byte.
An audit CSV records every duplicate decision.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

C13_C12_DIFF = 1.00335483507

TITLE_RE = re.compile(
    r"^(?P<prefix>.+)\.(?P<scan1>\d+)\.(?P<scan2>\d+)\.(?P<charge>\d+)(?:\s|$)"
)
CHARGE_RE = re.compile(r"^(?P<charge>\d+)\+$")


@dataclass
class Record:
    index: int
    start: int
    end: int
    title: str
    scan: int
    charge: int
    pepmass: float
    rt: float
    peak_count: int
    peak_hash: str


def is_peak_line(line: bytes) -> bool:
    """True when a stripped line starts with numeric m/z and intensity columns."""
    try:
        parts = line.split()
        if len(parts) < 2:
            return False
        float(parts[0])
        float(parts[1])
        return True
    except (ValueError, OverflowError):
        return False


def parse_mgf(path: Path):
    records = []
    first_block_start = None
    mass_monoisotopic = False

    in_block = False

    with path.open("rb") as fh:
        while True:
            line_start = fh.tell()
            raw = fh.readline()
            if not raw:
                break

            stripped = raw.strip()
            text = stripped.decode("utf-8", errors="replace")

            if not in_block:
                if text == "MASS=Monoisotopic":
                    mass_monoisotopic = True

                if text == "BEGIN IONS":
                    if first_block_start is None:
                        first_block_start = line_start

                    in_block = True
                    start = line_start
                    title = None
                    scan = None
                    charge = None
                    pepmass = None
                    rt = None
                    peak_count = 0
                    peak_hasher = hashlib.sha256()

                continue

            if text.startswith("TITLE="):
                title = text[6:]

            elif text.startswith("PEPMASS="):
                try:
                    pepmass = float(text[8:].split()[0])
                except ValueError as exc:
                    raise ValueError(
                        f"{path.name}: invalid PEPMASS: {text}"
                    ) from exc

            elif text.startswith("CHARGE="):
                value = text[7:].strip()
                match = CHARGE_RE.fullmatch(value)
                if not match:
                    raise ValueError(
                        f"{path.name}: expected one charge such as CHARGE=3+, got: {text}"
                    )
                charge = int(match.group("charge"))

            elif text.startswith("SCANS="):
                try:
                    scan = int(text[6:].strip())
                except ValueError as exc:
                    raise ValueError(
                        f"{path.name}: invalid SCANS: {text}"
                    ) from exc

            elif text.startswith("RTINSECONDS="):
                try:
                    rt = float(text[12:].strip())
                except ValueError as exc:
                    raise ValueError(
                        f"{path.name}: invalid RTINSECONDS: {text}"
                    ) from exc

            elif text == "END IONS":
                end = fh.tell()

                missing = [
                    name
                    for name, value in (
                        ("TITLE", title),
                        ("PEPMASS", pepmass),
                        ("CHARGE", charge),
                        ("SCANS", scan),
                        ("RTINSECONDS", rt),
                    )
                    if value is None
                ]

                if missing:
                    raise ValueError(
                        f"{path.name}: record {len(records)+1} missing "
                        + ", ".join(missing)
                    )

                if peak_count == 0:
                    raise ValueError(
                        f"{path.name}: record {len(records)+1} has zero MS2 peaks"
                    )

                # Validate this study's title convention:
                # No_07.10666.10666.2 (intensity=...)
                tm = TITLE_RE.match(title)
                if not tm:
                    raise ValueError(
                        f"{path.name}: unexpected TITLE format:\n  {title}"
                    )

                title_scan1 = int(tm.group("scan1"))
                title_scan2 = int(tm.group("scan2"))
                title_charge = int(tm.group("charge"))

                if title_scan1 != scan or title_scan2 != scan:
                    raise ValueError(
                        f"{path.name}: TITLE/SCANS mismatch:\n"
                        f"  TITLE={title}\n  SCANS={scan}"
                    )

                if title_charge != charge:
                    raise ValueError(
                        f"{path.name}: TITLE/CHARGE mismatch:\n"
                        f"  TITLE={title}\n  CHARGE={charge}+"
                    )

                records.append(
                    Record(
                        index=len(records),
                        start=start,
                        end=end,
                        title=title,
                        scan=scan,
                        charge=charge,
                        pepmass=pepmass,
                        rt=rt,
                        peak_count=peak_count,
                        peak_hash=peak_hasher.hexdigest(),
                    )
                )

                in_block = False
                continue

            elif is_peak_line(stripped):
                peak_count += 1
                # Same normalized fragment-list hashing used in the manual check.
                peak_hasher.update(stripped + b"\n")

    if in_block:
        raise ValueError(f"{path.name}: file ended before END IONS")

    if not records:
        raise ValueError(f"{path.name}: no spectra found")

    with path.open("rb") as fh:
        preamble = fh.read(first_block_start)

    return preamble, records, mass_monoisotopic


def find_isotope_relation(records, isotope_tol_mz, max_isotope_offset):
    """Check whether same-charge PEPMASS values differ by integer isotope spacing."""
    if len(records) < 2:
        return False, ""

    z = records[0].charge
    if any(r.charge != z for r in records):
        return False, ""

    spacing = C13_C12_DIFF / z
    ordered = sorted(records, key=lambda r: r.pepmass)

    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            delta = ordered[j].pepmass - ordered[i].pepmass

            for n in range(1, max_isotope_offset + 1):
                expected = n * spacing
                error = abs(delta - expected)

                if error <= isotope_tol_mz:
                    reason = (
                        f"isotope-like delta_mz={delta:.6f}; "
                        f"~{n}*(1.00335483507/{z})={expected:.6f}; "
                        f"error={error:.6f}"
                    )
                    return True, reason

    return False, ""


def choose_record(group, isotope_tol_mz, max_isotope_offset):
    """
    Resolve one duplicated-SCANS group whose MS2 peak lists are identical.
    """
    charges = sorted({r.charge for r in group})
    highest_charge = max(charges)
    candidates = [r for r in group if r.charge == highest_charge]

    reasons = ["identical MS2 peak list"]

    if len(charges) > 1:
        reasons.append(
            f"conflicting charges {','.join(map(str, charges))}; "
            f"kept highest charge {highest_charge}+"
        )

    if len(candidates) == 1:
        return candidates[0], "; ".join(reasons)

    isotope_like, isotope_reason = find_isotope_relation(
        candidates,
        isotope_tol_mz=isotope_tol_mz,
        max_isotope_offset=max_isotope_offset,
    )

    if isotope_like:
        selected = min(candidates, key=lambda r: (r.pepmass, r.index))
        reasons.append(isotope_reason)
        reasons.append("kept lowest precursor m/z")
        return selected, "; ".join(reasons)

    # Provider's study-specific rule for otherwise equivalent same-charge records.
    selected = min(candidates, key=lambda r: r.index)
    reasons.append(
        "same charge; non-isotope-scale m/z difference; kept first file record"
    )
    return selected, "; ".join(reasons)


def clean_file(
    input_path: Path,
    output_path: Path,
    report_path: Path,
    isotope_tol_mz: float,
    max_isotope_offset: int,
    overwrite: bool,
    dry_run: bool,
):
    if not dry_run:
        for p in (output_path, report_path):
            if p.exists() and not overwrite:
                raise FileExistsError(
                    f"Output exists: {p}\n"
                    f"Use --overwrite only if you intend to replace it."
                )

    preamble, records, mass_monoisotopic = parse_mgf(input_path)

    groups = defaultdict(list)
    for record in records:
        groups[record.scan].append(record)

    keep_indices = set()
    audit = []

    duplicate_groups = 0
    conflicting_charge_groups = 0
    isotope_groups = 0
    removed = 0

    for scan, group in sorted(groups.items()):
        if len(group) == 1:
            keep_indices.add(group[0].index)
            continue

        duplicate_groups += 1

        # Critical safety rule: only collapse duplicate scan IDs when
        # the fragment spectrum is demonstrably identical.
        signatures = {(r.peak_count, r.peak_hash) for r in group}

        if len(signatures) != 1:
            details = "\n".join(
                f"  record={r.index+1}, z={r.charge}, "
                f"PEPMASS={r.pepmass:.9f}, RT={r.rt:.3f}, "
                f"peaks={r.peak_count}, hash={r.peak_hash[:12]}..."
                for r in group
            )
            raise RuntimeError(
                f"{input_path.name}: SCANS={scan} occurs {len(group)} times "
                f"but the MS2 peak lists differ.\n"
                f"This is outside the validated duplicate pattern for this study.\n"
                f"No cleaned file should be used until this is reviewed:\n{details}"
            )

        if len({r.charge for r in group}) > 1:
            conflicting_charge_groups += 1

        selected, reason = choose_record(
            group,
            isotope_tol_mz=isotope_tol_mz,
            max_isotope_offset=max_isotope_offset,
        )

        if "isotope-like" in reason:
            isotope_groups += 1

        keep_indices.add(selected.index)
        removed += len(group) - 1

        for r in sorted(group, key=lambda x: x.index):
            audit.append(
                {
                    "input_file": input_path.name,
                    "scan": scan,
                    "record_number": r.index + 1,
                    "decision": "KEEP" if r.index == selected.index else "DROP",
                    "reason": reason,
                    "title": r.title,
                    "charge": r.charge,
                    "pepmass": f"{r.pepmass:.12f}",
                    "rt_seconds": f"{r.rt:.6f}",
                    "peak_count": r.peak_count,
                    "peak_sha256": r.peak_hash,
                    "selected_record_number": selected.index + 1,
                    "selected_title": selected.title,
                    "selected_charge": selected.charge,
                    "selected_pepmass": f"{selected.pepmass:.12f}",
                    "selected_rt_seconds": f"{selected.rt:.6f}",
                }
            )

    kept = [r for r in records if r.index in keep_indices]

    # Final integrity checks.
    duplicate_titles = [
        title
        for title, count in Counter(r.title for r in kept).items()
        if count > 1
    ]
    if duplicate_titles:
        raise RuntimeError(
            f"{input_path.name}: duplicate full TITLE values remain after cleaning:\n"
            + "\n".join(f"  {x}" for x in duplicate_titles[:10])
        )

    duplicate_scans_after = [
        scan
        for scan, count in Counter(r.scan for r in kept).items()
        if count > 1
    ]
    if duplicate_scans_after:
        raise RuntimeError(
            f"{input_path.name}: duplicate SCANS remain after cleaning: "
            + ", ".join(map(str, duplicate_scans_after[:20]))
        )

    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Preserve selected blocks byte-for-byte.
        with input_path.open("rb") as src, output_path.open("wb") as dst:
            dst.write(preamble)
            for r in kept:
                src.seek(r.start)
                dst.write(src.read(r.end - r.start))

        report_path.parent.mkdir(parents=True, exist_ok=True)

        fields = [
            "input_file",
            "scan",
            "record_number",
            "decision",
            "reason",
            "title",
            "charge",
            "pepmass",
            "rt_seconds",
            "peak_count",
            "peak_sha256",
            "selected_record_number",
            "selected_title",
            "selected_charge",
            "selected_pepmass",
            "selected_rt_seconds",
        ]

        with report_path.open("w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(audit)

    before_charge = Counter(r.charge for r in records)
    after_charge = Counter(r.charge for r in kept)

    print(f"\n=== {input_path.name} ===")
    print(f"Input spectra:                 {len(records)}")
    print(f"Unique SCANS before cleaning:  {len(groups)}")
    print(f"Duplicate SCANS groups:        {duplicate_groups}")
    print(f"Conflicting-charge groups:     {conflicting_charge_groups}")
    print(f"Isotope-like m/z groups:       {isotope_groups}")
    print(f"Records removed:               {removed}")
    print(f"Output spectra:                {len(kept)}")
    print(f"Unique SCANS after cleaning:   {len(set(r.scan for r in kept))}")
    print(
        f"MASS=Monoisotopic header:      "
        f"{'YES' if mass_monoisotopic else 'NO (warning)'}"
    )
    print(
        "Charge counts before:          "
        + ", ".join(f"{z}+={before_charge[z]}" for z in sorted(before_charge))
    )
    print(
        "Charge counts after:           "
        + ", ".join(f"{z}+={after_charge[z]}" for z in sorted(after_charge))
    )

    if dry_run:
        print("DRY RUN: no output files written.")
    else:
        print(f"Output MGF:                    {output_path}")
        print(f"Audit CSV:                     {report_path}")


def collect_inputs(path: Path):
    if path.is_file():
        if path.suffix.lower() != ".mgf":
            raise ValueError(f"Input is not an .mgf file: {path}")
        return [path]

    if path.is_dir():
        files = sorted(
            p
            for p in path.iterdir()
            if p.is_file()
            and p.suffix.lower() == ".mgf"
            and not p.name.lower().endswith("_plink3_ready.mgf")
        )
        if not files:
            raise ValueError(f"No .mgf files found in: {path}")
        return files

    raise FileNotFoundError(f"Input path does not exist: {path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Study-specific duplicate cleaner for Waters/ProteoWizard "
            "DDA crosslinking MGF files."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="One .mgf file or a directory containing .mgf files.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Directory for cleaned MGF and audit CSV files.",
    )
    parser.add_argument(
        "--isotope-tol-mz",
        type=float,
        default=0.01,
        help="Absolute tolerance for isotope-offset recognition (default 0.01 m/z).",
    )
    parser.add_argument(
        "--max-isotope-offset",
        type=int,
        default=4,
        help="Maximum isotope offset tested (default 4).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze without writing files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing outputs.",
    )
    args = parser.parse_args()

    try:
        files = collect_inputs(args.input)
        args.outdir.mkdir(parents=True, exist_ok=True)

        for input_path in files:
            output_path = args.outdir / f"{input_path.stem}_plink3_ready.mgf"
            report_path = args.outdir / f"{input_path.stem}_dedup_report.csv"

            clean_file(
                input_path=input_path,
                output_path=output_path,
                report_path=report_path,
                isotope_tol_mz=args.isotope_tol_mz,
                max_isotope_offset=args.max_isotope_offset,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )

        return 0

    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
