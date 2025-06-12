input_file = "metadata.csv"
output_file = "data/output.txt"

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        parts = line.strip().split("|")
        if len(parts) == 3:
            text = parts[2]
            outfile.write(text + "\n")

print("Conversion complete! Output written to", output_file)
