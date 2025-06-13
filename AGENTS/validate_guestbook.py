from __future__ import annotations

try:
    import os
    import re
    import sys
except Exception:
    import os
    import sys
    try:
        ENV_SETUP_BOX = os.environ["ENV_SETUP_BOX"]
    except KeyError as exc:
        raise RuntimeError("environment not initialized") from exc
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---

REPORTS_DIR = os.path.join(os.path.dirname(__file__), 'experience_reports')
TEMPLATES = {
    'template_experience_report.md',
    'template_doc_report.md',
    'template_tticket_report.md',
    'template_audit_report.md',
}
ARCHIVE_DIR = os.path.join(REPORTS_DIR, 'archive')
STICKIES_FILE = os.path.join(REPORTS_DIR, 'stickies.txt')
PATTERN = re.compile(
    r'(?:\d{4}-\d{2}-\d{2}|\d{10})_(DOC|TTICKET|AUDIT)_[A-Za-z0-9_]+\.md'
)


def sanitize(name):
    stem = os.path.splitext(name)[0]
    stem = re.sub(r'[^A-Za-z0-9]+', '_', stem)
    stem = re.sub(r'_+', '_', stem).strip('_')
    return f'0000000000_DOC_{stem}.md'

def validate_and_fix(interactive: bool = False):
    """Validate filenames in the guestbook folder.

    Parameters
    ----------
    interactive : bool, optional
        If True, prompt the user before fixing invalid filenames and dump the
        file contents so the user can decide how to proceed.  When False the
        function behaves like the old implementation and renames files
        automatically.
    """

    changed = False
    for fname in os.listdir(REPORTS_DIR):
        if fname in TEMPLATES or fname == 'AGENTS.md' or not fname.endswith('.md'):
            continue
        if PATTERN.fullmatch(fname):
            continue

        src = os.path.join(REPORTS_DIR, fname)
        new_name = sanitize(fname)

        if interactive:
            with open(src, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"Invalid filename: {fname}")
            print("----- file contents begin -----")
            print(content)
            print("----- file contents end -----")
            resp = input(
                f"Rename to {new_name}? [y]/n/d (dump to AGENTS)/s(skip): "
            ).strip().lower() or 'y'

            if resp.startswith('d'):
                dest = os.path.join(os.path.dirname(REPORTS_DIR), new_name)
                os.rename(src, dest)
                print(f"Moved {fname} -> {dest}")
                changed = True
                continue
            if resp.startswith('s'):
                print(f"Skipped {fname}")
                continue
            if not resp.startswith('n'):
                os.rename(src, os.path.join(REPORTS_DIR, new_name))
                print(f"Renamed {fname} -> {new_name}")
                changed = True
        else:
            os.rename(src, os.path.join(REPORTS_DIR, new_name))
            print(f"Renamed {fname} -> {new_name}")
            changed = True

    return changed


def load_stickies():
    if not os.path.isfile(STICKIES_FILE):
        return set()
    with open(STICKIES_FILE, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith('#')]
    return set(lines)


def archive_old_reports():
    from datetime import datetime

    def extract_epoch(fname: str) -> int:
        prefix = fname.split('_', 1)[0]
        if '-' in prefix:
            try:
                dt = datetime.strptime(prefix, "%Y-%m-%d")
                return int(dt.timestamp())
            except Exception:
                return 0
        try:
            return int(prefix)
        except ValueError:
            return 0

    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    stickies = load_stickies()
    files = [
        f
        for f in os.listdir(REPORTS_DIR)
        if f.endswith('.md') and f not in TEMPLATES and f != 'AGENTS.md'
    ]
    files.sort(key=extract_epoch)
    keep = stickies.union(files[-10:])
    for fname in files:
        if fname not in keep:
            src = os.path.join(REPORTS_DIR, fname)
            dest = os.path.join(ARCHIVE_DIR, fname)
            if not os.path.exists(dest):
                os.rename(src, dest)
                print(f'Archived {fname}')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Validate guestbook reports')
    parser.add_argument('--interactive', action='store_true',
                        help='Prompt before renaming or moving files')
    args = parser.parse_args()

    if not os.path.isdir(REPORTS_DIR):
        print('reports directory not found')
        raise SystemExit(1)

    changed = validate_and_fix(interactive=args.interactive)
    archive_old_reports()
    files = sorted(os.listdir(REPORTS_DIR))
    print('Current files:')
    for f in files:
        print(' -', f)
    if not changed:
        print('All filenames conform to pattern.')

