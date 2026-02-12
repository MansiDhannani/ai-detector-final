59.# Merge Multiple CSV Files
import csv

def merge_csv_files(output_file, *input_files):
    with open(output_file, 'w', newline='') as out:
        writer = None
        for file in input_files:
            with open(file, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)
                if writer is None:
                    writer = csv.writer(out)
                    writer.writerow(header)
                for row in reader:
                    writer.writerow(row)