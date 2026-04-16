from pathlib import Path

OUTPUT_FILE = "combined_output.txt"

def is_text_file(file_path: Path) -> bool:
    # Basic filter: skip directories and the output file itself
    return file_path.is_file() and file_path.name != OUTPUT_FILE

def read_and_append_files():
    current_dir = Path("/root/MLOPS-Experiment02")
    
    with open(OUTPUT_FILE, "a", encoding="utf-8") as out:
        for file_path in current_dir.iterdir():
            if not is_text_file(file_path):
                continue

            try:
                content = file_path.read_text(encoding="utf-8")
            except Exception as e:
                content = f"[Could not read file: {e}]"

            out.write(f'"{file_path.name}"\n')
            out.write(content)
            out.write("\n\n")  # separator between files

if __name__ == "__main__":
    read_and_append_files()
