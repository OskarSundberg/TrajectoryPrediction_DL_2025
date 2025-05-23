import os
import glob
import re
import statistics

def compute_average_metrics(
    model="SEASTAR",
    location="Valhallavagen",
    output_file="average_metrics_summary.txt",
    precision=6
):
    # Build directory path based on model and location
    directory = f"./data/Results/{model}/{location}"

    # Define the metrics of interest
    metric_names = ['Mean ADE', 'Mean FDE', 'Min ADE', 'Min FDE']

    # Find all *_metrics.txt files in the directory
    pattern = os.path.join(directory, "*_metrics.txt")
    txt_files = glob.glob(pattern)

    if not txt_files:
        print(f"No .txt files found in {directory}")
        return

    # Regular expressions to capture the metric values
    regexes = {name: re.compile(rf"{name}:\s*([0-9]+\.?[0-9]*)") for name in metric_names}

    # Store per-file metrics and accumulate values
    per_file_metrics = []
    metrics_accum = {name: [] for name in metric_names}

    # Parse each file
    for filepath in txt_files:
        file_metrics = {'filename': os.path.basename(filepath)}
        with open(filepath, 'r') as file:
            content = file.read()
            for name, regex in regexes.items():
                match = regex.search(content)
                if match:
                    value = float(match.group(1))
                    file_metrics[name] = value
                    metrics_accum[name].append(value)
                else:
                    file_metrics[name] = None
                    print(f"Warning: '{name}' not found in {file_metrics['filename']}")
        per_file_metrics.append(file_metrics)

    # Prepare result lines
    results = ["Per-file Metrics:"]
    for fm in per_file_metrics:
        values_str = ", ".join(
            f"{name}: {fm[name]:.{precision}f}" if fm[name] is not None else f"{name}: N/A"
            for name in metric_names
        )
        results.append(f"{fm['filename']}: {values_str}")

    # Compute and append statistics
    results.append("\nSummary Statistics Across Files:")
    for name, values in metrics_accum.items():
        if values:
            mean_val = statistics.mean(values)
            var_val = statistics.pvariance(values)
            std_val = statistics.pstdev(values)
            results.append(
                f"{name}: mean={mean_val:.{precision}f}, variance={var_val:.{precision}f}, stdev={std_val:.{precision}f} (from {len(values)} files)"
            )
        else:
            results.append(f"{name}: No values found.")

    # Print to console
    for line in results:
        print(line)
    print()

    # Compute full path for output
    output_path = os.path.join(directory, output_file)

    # Save to output file
    try:
        with open(output_path, 'w') as f:
            for line in results:
                f.write(line + "\n")
        print(f"Results written to {output_path}")
    except IOError as e:
        print(f"Failed to write to {output_path}: {e}")

if __name__ == "__main__":
    compute_average_metrics(model="SEASTAR", precision=2)
