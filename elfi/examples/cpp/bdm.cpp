#include <iostream>
#include <random>
#include <algorithm>
#include <iterator>
#include <cstring>
#include <fstream>


/**
Birth-Death-Mutation (BDM) process as explained in Tanaka et. al. 2006 [1].

References
----------
[1] Tanaka, Mark M., et al. "Using approximate Bayesian computation to estimate
tuberculosis transmission parameters from genotype data."
Genetics 173.3 (2006): 1511-1520.

Implemented for ELFI by Jarno Lintusaari, 2017.
*/


typedef unsigned int uint;


class BDM {
private:
    std::mt19937 gen;
    std::uniform_real_distribution<double> unif;


    template<typename T>
    uint draw_index_fast(const std::vector<T>& masses, T masses_total, size_t masses_end=-1) {
        if (masses_end == -1)
            masses_end = masses.size();

        // Draw the event index.
        float u = unif(gen)*masses_total;
        T masses_cum = T();
        for (size_t i=0; i < masses_end; i++) {
            masses_cum += masses[i];
            if ((float)masses_cum > u)
                return i;
        }
    }


public:


    int stopping_mode;


    BDM(uint seed, int stopping_mode=0) : gen(seed), unif(.0, 1.0), stopping_mode(stopping_mode) {}
        /**
         *
         * Parameters
         * ----------
         *
         * seed
         *     Seed for the random number generator
         * stopping_mode
         *     0: Stop immediately when arriving to the population size N (Tanaka et al. 2006)
         *     1: Stop just before the population would exceed the size N (Stadler 2011)
         */

    uint draw_event_index(const std::vector<float> event_rates, float rates_total) {
        return draw_index_fast(event_rates, rates_total);
    }


    uint draw_cluster_index(const std::vector<uint> clusters, uint pop_size, size_t cluster_end) {
        return draw_index_fast(clusters, pop_size, cluster_end);
    }


    std::vector<uint> simulate_population(float alpha, float beta, float theta, uint N) {
        /**
        Simulate a population from the BDM model.

        Parameters
        ----------
        alpha : float
            Birth rate
        delta : float
            Death rate
        theta : float
            Mutation rate
        N : int
            Size of the population to simulate

        Returns
        -------
        clusters
            Vector of clusters

        */

        // Set the event rates
        std::vector<float> rates {alpha, beta, theta};
        float rates_total = std::accumulate(rates.begin(), rates.end(), 0.0);

        // Set the initial clusters
        std::vector<uint> clusters(N, 0);
        uint pop_size = 1;
        clusters[0] = 1;
        // Keep track of the last cluster index to optimize the drawing of the index
        size_t cluster_end = 1;

        if (stopping_mode==1) {
            N += 1;
        }

        uint event;
        uint cluster;

        while (pop_size < N && pop_size > 0) {
            // Draw the event
            event = draw_event_index(rates, rates_total);
            cluster = draw_cluster_index(clusters, pop_size, cluster_end);

            if (event==0) {
                clusters[cluster] += 1;
                pop_size += 1;
            }
            else if (event==1) {
                clusters[cluster] -= 1;
                pop_size -= 1;
            }
            else if (event==2 && clusters[cluster] > 1) {
                // If there is only one member in this cluster, we do not need to do anything
                clusters[cluster] -= 1;

                // Find an empty place and begin a new cluster
                for (auto i = clusters.begin(); i != clusters.end(); ++i) {
                    if (*i == 0) {
                        *i = 1;
                        cluster_end = std::max(cluster_end, (size_t)(i-clusters.begin()) + 1);
                        break;
                    }
                }
            }
        }

        if (event==0 && stopping_mode==1) {
            clusters[cluster] -= 1;
            N -= 1;
            pop_size -= 1;
        }

        return clusters;
    }

};


/************************************************* Interface **********************************************************/


void write_population(std::vector<uint>& pop) {
    auto i = pop.begin();
    for (i; i != pop.end()-1; ++i) std::cout << *i << ' ';
    std::cout << *i;
}


void run_from_file(std::string file, BDM& bdm) {
    std::ifstream infile(file);

    float alpha, delta, theta;
    uint N;

    while (infile >> alpha >> delta >> theta >> N)
    {
        std::vector<uint> pop = bdm.simulate_population(alpha, delta, theta, N);
        write_population(pop);
        std::cout << "\n";
    }
    return;
}


void run_from_args(float alpha, float delta, float theta, uint N, BDM& bdm) {
    // Construct the simulator and simulate a population
    std::vector<uint> pop = bdm.simulate_population(alpha, delta, theta, N);
    // Write to stdout
    write_population(pop);
    return;
}


int parse_seed(int argc, char* argv[], u_int32_t &seed) {
    for (int i=1; i < argc; i++) {
        if (strcmp(argv[i], "--seed") == 0 && argc >= i+2) {
            seed = std::stoul(argv[i+1]);
            return i;
        }
    }
    return -1;
}


int parse_mode(int argc, char* argv[], int &mode) {
    for (int i=1; i < argc; i++) {
        if (strcmp(argv[i], "--mode") == 0 && argc >= i+2) {
            mode = std::stoi(argv[i+1]);
            return i;
        }
    }
    return -1;
}


void print_usage() {
    // Inform the user how to use the program
    std::cout << "\nUsage is: bdm <alpha> <delta> <theta> <N> [--seed <seed>] [--mode <mode>]\n";
    std::cout << "      or: bdm input_file [--seed <seed>] [--mode <mode>]\n";
}


int main(int argc, char* argv[]) {

    if (argc < 4) { print_usage(); return 0; }

    int num_positional_args = argc;

    // Parse keyword arguments
    u_int32_t seed = (u_int32_t) time(0);
    if (parse_seed(argc, argv, seed) > -1) num_positional_args -= 2;

    int stopping_mode = 0;
    if (parse_mode(argc, argv, stopping_mode) > -1) num_positional_args -= 2;

    BDM bdm(seed, stopping_mode);

    // Run with the positional arguments
    if (num_positional_args == 2) {
        // Input file
        run_from_file(argv[1], bdm);
    }
    else if (num_positional_args == 5) {
        float alpha = std::strtof(argv[1], NULL);
        float delta = std::strtof(argv[2], NULL);
        float theta = std::strtof(argv[3], NULL);
        uint N = (uint) std::stoul(argv[4], NULL);
        run_from_args(alpha, delta, theta, N, bdm);
    }
    else {
        std::cout << "Could not interpret the input: ";
        for (int i = 0; i < argc; i++) {
            std::cout << argv[i] << ' ';
        }
        print_usage();
        return -1;
    }

    return 0;
}
