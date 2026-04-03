#!/bin/bash

# Output file - just the list of files for now
OUTPUT="file_list.txt"

# Clear or create the output file
> "$OUTPUT"

echo "=== Finding all files recursively (excluding noise) ==="

# Find all files, excluding common unwanted directories and files
find . -type f \
    ! -path '*/venv/*' \
    ! -path '*/__pycache__/*' \
    ! -path '*/.git/*' \
    ! -path '*/artifacts/*' \
    ! -path '*/logs/*' \
    ! -path './Allfiles' \
    ! -path './combined*.txt' \
    ! -path './file_list.txt' \
    ! -name '*.pyc' \
    ! -name '*.log' \
    ! -name 'combined_python_files.txt' \
    ! -name 'combine_all_files.sh' \
    ! -name 'file_list.txt' \
    -print0 | sort -z | while IFS= read -r -d '' file; do

    # Print relative path (cleaner output)
    echo "${file#./}" >> "$OUTPUT"

done

echo "Done! File list created: $OUTPUT"
echo "Total files found: $(wc -l < "$OUTPUT")"
echo ""
echo "You can view the list with:"
echo "cat $OUTPUT | less"