import subprocess
import os
import time
import argparse
import re
import sys

def main():
    """
    This script automates the evaluation of the RNN model with different n-gram language models.
    It starts the Redis server, then for each specified language model, it:
    1. Starts the language model as a subprocess.
    2. Runs the evaluate_model.py script.
    3. Parses the output to extract the Word Error Rate (WER).
    4. Terminates the language model subprocess.
    Finally, it prints a summary table of the WER for each language model and shuts down the Redis server.
    """
    parser = argparse.ArgumentParser(description='Run evaluation with different language models and compare WER.')
    parser.add_argument('--lm_names', nargs='+', default=['1gram'], choices=['1gram', '3gram', '5gram'],
                        help='A list of n-gram language models to evaluate (e.g., 1gram 3gram).')
    parser.add_argument('--model_path', type=str, default='../data/t15_pretrained_rnn_baseline',
                        help='Path to the pretrained model directory.')
    parser.add_argument('--data_dir', type=str, default='../data/hdf5_data_final',
                        help='Path to the dataset directory.')
    parser.add_argument('--rnn_gpu', type=int, default=1,
                        help='GPU number for RNN model inference.')
    parser.add_argument('--lm_gpu', type=int, default=0,
                        help='GPU number for language model inference.')

    args = parser.parse_args()

    # --- Start Redis Server ---
    print("Starting Redis server...")
    redis_process = subprocess.Popen(['redis-server', '--daemonize', 'yes'])
    # It's better to check if redis is up, but for this script, a short sleep is acceptable
    # since it's a local server and should start quickly.
    time.sleep(2)
    print("Redis server started.")

    results = {}

    try:
        for lm_name in args.lm_names:
            print(f"--- Evaluating with {lm_name} language model ---")

            # --- Start Language Model ---
            lm_process = start_language_model(lm_name, args.lm_gpu)
            if lm_process is None:
                print(f"Skipping {lm_name} due to previous error.")
                continue

            # --- Run Evaluation ---
            wer = run_evaluation(args.model_path, args.data_dir, args.rnn_gpu)
            if wer is not None:
                results[lm_name] = wer

            # --- Stop Language Model ---
            print("Stopping language model...")
            lm_process.terminate()
            try:
                lm_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                lm_process.kill()
            print("Language model stopped.")

    finally:
        # --- Stop Redis Server ---
        print("Stopping Redis server...")
        subprocess.run(['redis-cli', 'shutdown'])
        print("Redis server stopped.")

    # --- Print Summary Table ---
    print("\n--- Evaluation Summary ---")
    if results:
        print(f"{'Language Model':<20} | {'Word Error Rate (%)':<20}")
        print("-" * 43)
        for lm_name, wer in results.items():
            print(f"{lm_name:<20} | {wer:<20.2f}")
    else:
        print("No evaluation results to display.")

def start_language_model(lm_name, lm_gpu):
    """Starts the language model as a subprocess and waits for it to be ready."""
    lm_path = f'../language_model/pretrained_language_models/openwebtext_{lm_name}_lm_sil'
    if not os.path.exists(lm_path):
        print(f"Warning: Language model path not found: {lm_path}. Skipping.")
        print("Please download the language models as described in the README.")
        return None

    lm_command = [
        'python',
        '../language_model/language-model-standalone.py',
        '--lm_path', lm_path,
        '--do_opt',
        '--nbest', '100',
        '--acoustic_scale', '0.325',
        '--blank_penalty', '90',
        '--alpha', '0.55',
        '--redis_ip', 'localhost',
        '--gpu_number', str(lm_gpu)
    ]

    print(f"Starting language model: {' '.join(lm_command)}")

    conda_path = os.environ.get('CONDA_EXE', '').replace('bin/conda', 'bin/activate')
    if not conda_path:
        print("Error: Conda executable not found. Please make sure conda is installed and CONDA_EXE is set.")
        return None

    lm_process = subprocess.Popen(
        ['bash', '-c', f'source {conda_path} b2txt25_lm && {" ".join(lm_command)}'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    # Wait for the language model to be ready
    print("Waiting for language model to initialize...")
    ready = False
    for line in iter(lm_process.stdout.readline, ''):
        print(f"[LM Log] {line.strip()}")
        if "Successfully connected to the redis server" in line:
            print("Language model is ready.")
            ready = True
            break
        if "Traceback" in line or "Error" in line:
            print("Error starting language model.")
            lm_process.terminate()
            return None

    if not ready:
        print("Error: Language model did not start successfully.")
        lm_process.terminate()
        return None

    return lm_process

def run_evaluation(model_path, data_dir, rnn_gpu):
    """Runs the evaluation script and parses the WER from its output."""
    eval_command = [
        'python',
        'evaluate_model.py',
        '--model_path', model_path,
        '--data_dir', data_dir,
        '--eval_type', 'val', # Always use val set for WER comparison
        '--gpu_number', str(rnn_gpu)
    ]
    print(f"Running evaluation: {' '.join(eval_command)}")

    # We need to run this in the `b2txt25` conda environment
    conda_path = os.environ.get('CONDA_EXE', '').replace('bin/conda', 'bin/activate')
    if not conda_path:
        print("Error: Conda executable not found. Please make sure conda is installed and CONDA_EXE is set.")
        return None

    eval_process = subprocess.Popen(
        ['bash', '-c', f'source {conda_path} b2txt25 && {" ".join(eval_command)}'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__)) # Run from the script's directory
    )

    wer = None
    output = []
    for line in iter(eval_process.stdout.readline, ''):
        sys.stdout.write(line)
        output.append(line)
        if "Aggregate Word Error Rate (WER):" in line:
            match = re.search(r'Aggregate Word Error Rate \(WER\): (\d+\.\d+)', line)
            if match:
                wer = float(match.group(1))

    eval_process.wait()
    if eval_process.returncode != 0:
        print("Error running evaluation script.")
        print("".join(output))
        return None

    return wer

if __name__ == '__main__':
    main()
