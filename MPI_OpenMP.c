#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <omp.h>
#include <mpi.h>

bool is_prime(long n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (long i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) return false;
    }
    return true;
}

int compare(const void *a, const void *b) {
    long arg1 = *(long*)a;
    long arg2 = *(long*)b;
    return (arg1 > arg2) - (arg1 < arg2);
}

int main(int argc, char *argv[]) {
    int provided;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0) printf("Usage: mpirun -np <nodes> %s <file.csv> <threads_per_node>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    const char* filename = argv[1];
    int num_threads = atoi(argv[2]);
    omp_set_num_threads(num_threads);

    long capacity = 10000000;
    long *data = malloc(capacity * sizeof(long));
    long count = 0;


    if (rank == 0) {
        FILE* file = fopen(filename, "r");
        if (file) {
            while (count < capacity && fscanf(file, "%ld,", &data[count]) != EOF) {
                count++;
            }
            fclose(file);
        }
    }

    // Broadcast the number of elements to all ranks
    MPI_Bcast(&count, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    if (count == 0) {
        if (rank == 0) printf("Error: No data loaded.\n");
        free(data);
        MPI_Finalize();
        return 1;
    }

    // Broadcast the actual array data
    MPI_Bcast(data, count, MPI_LONG, 0, MPI_COMM_WORLD);

    // All ranks sort the data (simplest way to ensure binary search works)
    qsort(data, count, sizeof(long), compare);

    // Divide the work: Calculate the range for this specific MPI rank
    long chunk_size = (count + size - 1) / size;
    long start_idx = rank * chunk_size;
    long end_idx = (start_idx + chunk_size < count) ? (start_idx + chunk_size) : count;

    long local_twin_primes = 0;

    // Parallel Search using OpenMP within this rank's assigned range
    #pragma omp parallel for reduction(+:local_twin_primes) schedule(dynamic, 1000)
    for (long i = start_idx; i < end_idx; i++) {
        if (i >= count - 1) continue;
        
        // Skip duplicates (including boundary check between ranks)
        if (i > 0 && data[i] == data[i-1]) continue;

        long p = data[i];
        if (is_prime(p)) {
            long target = p + 2;
            long low = 0, high = count - 1; // Search the full array for the twin
            
            while (low <= high) {
                long mid = low + (high - low) / 2;
                if (data[mid] == target) {
                    if (is_prime(target)) {
                        local_twin_primes++;
                    }
                    break;
                }
                if (data[mid] < target) low = mid + 1;
                else high = mid - 1;
            }
        }
    }

    // Reduce all local counts to the global count at Rank 0
    long global_twin_primes = 0;
    MPI_Reduce(&local_twin_primes, &global_twin_primes, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("MPI Processes: %d | Threads per Process: %d\n", size, num_threads);
        printf("Total elements: %ld\n", count);
        printf("Unique twin prime pairs: %ld\n", global_twin_primes);
    }

    free(data);
    MPI_Finalize();
    return 0;
}
