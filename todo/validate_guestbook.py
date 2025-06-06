import os
import re

REPORTS_DIR = os.path.join(os.path.dirname(__file__), 'experience_reports')
TEMPLATE = 'template_experience_report.md'
PATTERN = re.compile(r'\d{4}-\d{2}-\d{2}_v\d+_[A-Za-z0-9_]+\.md')

def sanitize(name):
    stem = os.path.splitext(name)[0]
    stem = re.sub(r'[^A-Za-z0-9]+', '_', stem)
    stem = re.sub(r'_+', '_', stem).strip('_')
    return f'0000-00-00_v0_{stem}.md'

def validate_and_fix():
    changed = False
    for fname in os.listdir(REPORTS_DIR):
        if fname == TEMPLATE or not fname.endswith('.md'):
            continue
        if not PATTERN.fullmatch(fname):
            new_name = sanitize(fname)
            os.rename(os.path.join(REPORTS_DIR, fname), os.path.join(REPORTS_DIR, new_name))
            print(f'Renamed {fname} -> {new_name}')
            changed = True
    return changed

if __name__ == '__main__':
    if not os.path.isdir(REPORTS_DIR):
        print('reports directory not found')
        raise SystemExit(1)
    changed = validate_and_fix()
    files = sorted(os.listdir(REPORTS_DIR))
    print('Current files:')
    for f in files:
        print(' -', f)
    if not changed:
        print('All filenames conform to pattern.')
