#!/bin/bash

# Input and output files
LIST_FILE="file_list.txt"
OUTPUT="Allfiles"

# Check if file_list.txt exists
if [ ! -f "$LIST_FILE" ]; then
    echo "Error: $LIST_FILE not found!"
    echo "Please make sure file_list.txt exists in the current directory."
    exit 1
fi

# Clear the output file
> "$OUTPUT"

echo "=== Combining files listed in $LIST_FILE into $OUTPUT ==="

# Read each file from the list and combine with header
while IFS= read -r file || [ -n "$file" ]; do
    # Skip empty lines and comments
    [[ -z "$file" || "$file" =~ ^# ]] && continue

    # Remove leading ./ if present for cleaner display
    display_name="${file#./}"

    if [ -f "$file" ]; then
        echo "=========================================" >> "$OUTPUT"
        echo "=== FILE: $display_name ===" >> "$OUTPUT"
        echo "=========================================" >> "$OUTPUT"
        echo "" >> "$OUTPUT"
        
        cat "$file" >> "$OUTPUT"
        
        echo "" >> "$OUTPUT"
        echo "=========================================" >> "$OUTPUT"
        echo "" >> "$OUTPUT"
        
        echo "Added: $display_name"
    else
        echo "Warning: File not found - $file" >&2
    fi
done < "$LIST_FILE"

echo ""
echo "Done! Combined file created: $OUTPUT"
echo "Total size: $(du -h "$OUTPUT" | cut -f1)"
echo "You can view it with: less $OUTPUT"
