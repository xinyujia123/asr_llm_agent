import re
import cn2an
from pathlib import Path

def clean_file(file_path, new_file_path):
    print(f"Processing: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Callback: Replace numbers with Chinese inside the matched string
        def convert_in_quote(match):
            return re.sub(r'\d+(\.\d+)?', lambda m: cn2an.an2cn(m.group()), match.group())

        # Match text inside Chinese quotes (“...”) or ASCII quotes ("...")
        # and apply number conversion only to those parts
        new_content = re.sub(r'“[^”]*”|"[^"]*"', convert_in_quote, content)

        with open(new_file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Done. Saved to: {new_file_path}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    clean_file('nurse_script.txt', 'nurse_script_tn.txt')