#!/usr/bin/env python
"""
Multi-node PCAP to Parquet: each MPI rank copies its share of PCAPs from
a central FS (e.g. pscratch) to /dev/shm with limited parallelism to avoid
stressing the FS, then runs tshark in parallel on /dev/shm and writes
one Parquet per rank to the output dir.
"""
import argparse
import os
import subprocess
from mpi4py import MPI
from multiprocessing import Pool, cpu_count
import pyarrow as pa
import pyarrow.parquet as pq
import logging
from concurrent.futures import ThreadPoolExecutor
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(process)d] %(message)s')

# Default: limit copy parallelism to avoid hammering centralized FS (e.g. pscratch)
DEFAULT_COPY_WORKERS = 8
# Default: use cpu_count() for tshark workers; can oversubscribe via CLI
DEFAULT_TSHARK_WORKERS = None


def process_pcap(file_path, fields, schema, use_tshark_quiet=True):
    logging.info(f"Starting process_pcap for {file_path}")
    try:
        command = [
            "tshark",
            "-r", file_path,
            "-T", "fields",
        ]
        if use_tshark_quiet:
            command.append("-q")
        for field in fields:
            command.extend(["-e", field])
        command.extend(["-E", "separator=,"])

        try:
            logging.info(f"Running tshark for {file_path}")
            result = subprocess.run(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
            )
            logging.info(f"Finished reading {file_path}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error processing {file_path}: {e}")
            logging.error(f"stderr: {e.stderr}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error with {file_path}: {e}")
            return None

        lines = result.stdout.strip().split("\n")
        data = []
        for line in lines:
            try:
                row = line.strip().split(',')
                src_port = int(row[2]) if len(row) > 2 and row[2].isdigit() else (int(row[4]) if len(row) > 4 and row[4].isdigit() else None)
                dst_port = int(row[3]) if len(row) > 3 and row[3].isdigit() else (int(row[5]) if len(row) > 5 and row[5].isdigit() else None)
                timestamp = float(row[6]) if len(row) > 6 else 0.0
                size = int(row[7]) if len(row) > 7 and row[7].isdigit() else 0
                data.append((
                    row[0] if len(row) > 0 else "",
                    row[1] if len(row) > 1 else "",
                    src_port,
                    dst_port,
                    timestamp,
                    size
                ))
            except (IndexError, ValueError, TypeError):
                continue

        if data:
            table = pa.Table.from_arrays(
                [list(column) for column in zip(*data)],
                schema=schema
            )
            return table
        return None
    except subprocess.CalledProcessError as e:
        logging.error(f"Error processing file {file_path}: {e}")
        return None


def save_to_parquet(tables, output_dir, rank):
    output_file = os.path.join(output_dir, f"output_node_{rank}.parquet")
    combined_table = pa.concat_tables(tables)
    pq.write_table(combined_table, output_file, compression="snappy")
    logging.info(f"Node {rank}: Parquet file saved to {output_file}")


def process_files_in_parallel(node_files, fields, schema, tshark_workers, use_tshark_quiet):
    n_workers = tshark_workers if tshark_workers is not None else cpu_count()
    logging.info(f"Starting process_files_in_parallel with {n_workers} workers")
    with Pool(n_workers) as pool:
        results = pool.starmap(
            process_pcap,
            [(f, fields, schema, use_tshark_quiet) for f in node_files]
        )
    return [t for t in results if t is not None]


def copy_files_parallel(source_files, target_files, num_workers=4):
    """Copy files from source to target with limited parallelism (e.g. to avoid stressing pscratch)."""
    def copy_file(src_dest):
        src, dest = src_dest
        shutil.copy(src, dest)
        logging.info(f"Copied {src} to {dest}")

    pairs = list(zip(source_files, target_files))
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(executor.map(copy_file, pairs))


def process_all_pcaps(pcap_dir, output_dir, copy_workers=DEFAULT_COPY_WORKERS,
                      tshark_workers=DEFAULT_TSHARK_WORKERS, use_tshark_quiet=True):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    fields = [
        "ip.src",
        "ip.dst",
        "tcp.srcport",
        "tcp.dstport",
        "udp.srcport",
        "udp.dstport",
        "frame.time_epoch",
        "frame.len"
    ]

    schema = pa.schema([
        ("src_ip", pa.string()),
        ("dst_ip", pa.string()),
        pa.field("src_port", pa.int32(), nullable=True),
        pa.field("dst_port", pa.int32(), nullable=True),
        ("timestamp", pa.float64()),
        ("size", pa.int32())
    ])

    all_filenames = [f for f in os.listdir(pcap_dir)]
    all_pcap_files = [os.path.join(pcap_dir, f) for f in all_filenames]

    temp_folder = "/dev/shm/files_to_process"
    temp_parquet_folder = "/dev/shm/parquet_out"
    os.makedirs(temp_folder, exist_ok=True)
    os.makedirs(temp_parquet_folder, exist_ok=True)

    copy_to_files = [os.path.join(temp_folder, f) for f in all_filenames]

    files_per_node = len(all_pcap_files) // size
    start_idx = rank * files_per_node
    end_idx = start_idx + files_per_node if rank != size - 1 else len(all_pcap_files)
    node_sources = all_pcap_files[start_idx:end_idx]
    node_targets = copy_to_files[start_idx:end_idx]
    node_files = copy_to_files[start_idx:end_idx]

    copy_files_parallel(node_sources, node_targets, num_workers=copy_workers)
    logging.info(f"Node {rank}: Copy complete. Processing {len(node_files)} files.")

    tables = process_files_in_parallel(
        node_files, fields, schema, tshark_workers, use_tshark_quiet
    )

    save_to_parquet(tables, temp_parquet_folder, rank)
    out_parquet = os.path.join(output_dir, f"output_node_{rank}.parquet")
    shutil.move(os.path.join(temp_parquet_folder, f"output_node_{rank}.parquet"), out_parquet)

    shutil.rmtree(temp_folder, ignore_errors=True)
    shutil.rmtree(temp_parquet_folder, ignore_errors=True)

    logging.info(f"Node {rank}: Processing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process PCAP files and save outputs as Parquet files (multi-node MPI)."
    )
    parser.add_argument("pcap_dir", type=str, help="Path to the directory containing input PCAP files.")
    parser.add_argument("output_dir", type=str, help="Path to the directory to save output Parquet files.")
    parser.add_argument(
        "--copy_workers",
        type=int,
        default=DEFAULT_COPY_WORKERS,
        help=f"Parallelism when copying PCAPs from central FS to /dev/shm (default {DEFAULT_COPY_WORKERS}); lower to reduce load on pscratch.",
    )
    parser.add_argument(
        "--tshark_workers",
        type=int,
        default=None,
        help="Number of parallel tshark processes per node (default: cpu_count()).",
    )
    parser.add_argument(
        "--no_tshark_quiet",
        action="store_true",
        help="Disable tshark -q (quiet) flag.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    process_all_pcaps(
        args.pcap_dir,
        args.output_dir,
        copy_workers=args.copy_workers,
        tshark_workers=args.tshark_workers,
        use_tshark_quiet=not args.no_tshark_quiet,
    )
