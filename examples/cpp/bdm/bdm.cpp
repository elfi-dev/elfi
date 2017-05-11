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


    BDM(uint seed) : gen(seed), unif(.0, 1.0) {}


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
            Size of the population

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

        while (pop_size < N && pop_size > 0) {
            // Draw the event
            uint event = draw_event_index(rates, rates_total);
            uint cluster = draw_cluster_index(clusters, pop_size, cluster_end);

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

        return clusters;
    }

};


/************************************************* Interface **********************************************************/


void write_population(std::vector<uint>& pop) {
    auto i = pop.begin();
    for (i; i != pop.end()-1; ++i) std::cout << *i << ' ';
    std::cout << *i;
}


void run_from_file(std::string file, u_int32_t seed) {
    std::ifstream infile(file);

    float alpha, delta, theta;
    uint N;

    BDM bdm(seed);

    while (infile >> alpha >> delta >> theta >> N)
    {
        std::vector<uint> pop = bdm.simulate_population(alpha, delta, theta, N);
        write_population(pop);
        std::cout << "\n";
    }
    return;
}


void run_from_args(float alpha, float delta, float theta, uint N, u_int32_t seed) {
    // Construct the simulator and simulate a population
    BDM bdm(seed);
    std::vector<uint> pop = bdm.simulate_population(alpha, delta, theta, N);
    // Write to stdout
    write_population(pop);
    return;
}


int parse_seed(int argc, char* argv[], u_int32_t &seed) {
    for (int i=1; i < argc; i++) {
        if (strcmp(argv[i], "--seed") == 0) {
            seed = std::stoul(argv[i+1]);
            return i;
        }
    }
    return -1;
}


int main(int argc, char* argv[]) {
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
            Size of the full population

        Returns
        -------
        clusters
            Vector of clusters

    */

    if (argc < 4) {
        // Inform the user how to use the program
        std::cout << "Usage is: bdm <alpha> <delta> <theta> <N> [--seed <seed>]\n";
        std::cout << "      or: bdm input_file [--seed <seed>]\n";
        return 0;
    }

    int num_positional_args = argc - 2;
    u_int32_t seed;
    if (parse_seed(argc, argv, seed) == -1) {
        num_positional_args = argc;
        seed = (u_int32_t) time(0);
    }

    if (num_positional_args == 2) {
        // Input file
        run_from_file(argv[1], seed);
    }
    else if (num_positional_args == 5) {
        float alpha = std::strtof(argv[1], NULL);
        float delta = std::strtof(argv[2], NULL);
        float theta = std::strtof(argv[3], NULL);
        uint N = (uint) std::stoul(argv[4], NULL);
        run_from_args(alpha, delta, theta, N, seed);
    }
    else {
        std::cout << "Could not interpret the input: ";
        for (int i = 0; i < argc; i++) {
            std::cout << argv[i] << ' ';
        }
        std::cout << ". See bdm --help\n\n";
        return -1;
    }

    return 0;
}
