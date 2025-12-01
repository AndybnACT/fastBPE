#!/bin/bash
#SBATCH -A m5101                 # NERSC account
#SBATCH -C cpu                   # Perlmutter CPU nodes
#SBATCH -q regular               # regular queue (up to 12h)
#SBATCH -t 01:00:00              # walltime
#SBATCH -J fastbpe_all           # job name
#SBATCH -N 1                     # one node
#SBATCH -c 128                   # 128 CPU cores (for OpenMP)
#SBATCH -o fastbpe_all.%j.out    # stdout
#SBATCH -e fastbpe_all.%j.err    # stderr

cd "$SLURM_SUBMIT_DIR"

# Reset Cray PE as recommended by system
source /opt/cray/pe/cpe/24.07/restore_lmod_system_defaults.sh

echo "Working directory: $(pwd)"
ls -lh Makefile fastBPE/main.cc fastBPE/fastBPE.hpp enwik9 codes.bpe vocab.fastBPE

THREADS=(1 2 4 8 32 64 128)

extract_bandwidth() {
    # Extract internal Time spent (ms) and Bandwidth (MB/sec) from the program output log.
    logfile="$1"

    # Time spent = 153139133ms
    time_ms=$(grep -m1 "Time spent" "$logfile" \
        | sed -E 's/.*Time spent *= *([0-9]+)ms.*/\1/')

    # Bandwidth = 6.53001e+06 B/sec, 6.2275MB/sec
    bw_MB=$(grep -m1 "Bandwidth" "$logfile" \
        | sed -E 's/.*Bandwidth *= *[^,]*, *([0-9.e+]+)MB\/sec.*/\1/')

    # Fallback if parsing fails
    if [ -z "$time_ms" ]; then
        time_ms="NA"
    fi
    if [ -z "$bw_MB" ]; then
        bw_MB="NA"
    fi

    echo "$time_ms" "$bw_MB"
}


############################################################
# Part 1 — GNU environment: build & run critical/merge     #
############################################################

echo
echo "========== Building GNU OpenMP variants (critical, merge) =========="

module purge
module load PrgEnv-gnu

echo "[GNU] Building..."
make omp_critical omp_merge

echo "[GNU] Binaries:"
ls -lh fast.omp_crt fast.omp_stm

export OMP_PROC_BIND=spread
export OMP_PLACES=cores

#######################
# Run fast.omp_crt    #
#######################
echo
echo "================= Running fast.omp_crt (GNU / OMP + CRITICAL) ================="
for t in "${THREADS[@]}"; do
    echo
    echo ">>> [fast.omp_crt] OMP_NUM_THREADS=$t"
    export OMP_NUM_THREADS=$t

    start=$(date +%s)

    logfile="crt_${t}.log"
    ./fast.omp_crt applybpe out_crt_gnu_${t}.txt enwik9 codes.bpe vocab.fastBPE \
        | tee "$logfile"

    end=$(date +%s)
    elapsed=$((end - start))

    read time_ms bw_MB < <(extract_bandwidth "$logfile")

    echo "[GNU][CRT] OMP_NUM_THREADS=$t  elapsed=${elapsed}s  internal_time=${time_ms}ms  bandwidth=${bw_MB}MB/s"
done

##############################
# Run fast.omp_stm           #
##############################
echo
echo "================= Running fast.omp_stm (GNU / OMP + STM MERGE) ================="
for t in "${THREADS[@]}"; do
    echo
    echo ">>> [fast.omp_stm] OMP_NUM_THREADS=$t"
    export OMP_NUM_THREADS=$t

    start=$(date +%s)

    logfile="stm_${t}.log"
    ./fast.omp_stm applybpe out_stm_gnu_${t}.txt enwik9 codes.bpe vocab.fastBPE \
        | tee "$logfile"

    end=$(date +%s)
    elapsed=$((end - start))

    read time_ms bw_MB < <(extract_bandwidth "$logfile")

    echo "[GNU][STM] OMP_NUM_THREADS=$t  elapsed=${elapsed}s  internal_time=${time_ms}ms  bandwidth=${bw_MB}MB/s"
done


############################################################
# Part 2 — Intel environment: build & run TBB version      #
############################################################

echo
echo "========== Building Intel TBB variant =========="

module intel
module load PrgEnv-intel
# If needed:
# module load intel-oneapi-tbb

make omp_tbb

echo "[Intel] Binaries:"
ls -lh fast.omp_tbb

export OMP_PROC_BIND=spread
export OMP_PLACES=cores

##############################
# Run fast.omp_tbb           #
##############################
echo
echo "================= Running fast.omp_tbb (Intel / OMP + TBB) ================="
for t in "${THREADS[@]}"; do
    echo
    echo ">>> [fast.omp_tbb] OMP_NUM_THREADS=$t"
    export OMP_NUM_THREADS=$t

    start=$(date +%s)

    logfile="tbb_${t}.log"
    ./fast.omp_tbb applybpe out_tbb_intel_${t}.txt enwik9 codes.bpe vocab.fastBPE \
        | tee "$logfile"

    end=$(date +%s)
    elapsed=$((end - start))

    read time_ms bw_MB < <(extract_bandwidth "$logfile")

    echo "[Intel][TBB] OMP_NUM_THREADS=$t  elapsed=${elapsed}s  internal_time=${time_ms}ms  bandwidth=${bw_MB}MB/s"
done

echo
echo "All runs (GNU critical/merge + Intel TBB) completed."
