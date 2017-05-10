#include <iostream>
#include <random>
#include <algorithm>
#include <iterator>
#include <cstring>

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

    std::vector<float> event_cum_pmf;
    std::vector<float>* cluster_cum_pmf;


    uint draw_index(const std::vector<float>& cum_pmf) {
        // Draw the event index.
        double u = unif(gen);
        for (uint i=0; i < cum_pmf.size(); i++)
            if (cum_pmf[i] > u)
                return i;
    }


    template<typename T>
    void update_cum_pmf(const std::vector<T>& masses, std::vector<float>* cum_pmf, uint masses_end=-1) {
        if (masses_end == -1)
            masses_end = masses.size();
        // Cumulative sums up to masses_end
        std::partial_sum(masses.begin(), masses.begin() + masses_end, cum_pmf->begin());
        // Normalize to pmf up to masses_end
        std::transform(cum_pmf->begin(), cum_pmf->begin() + masses_end, cum_pmf->begin(), std::bind2nd(std::divides<float>(), cum_pmf->at(masses_end-1)));
    }


public:


    BDM(uint seed) : gen(seed), unif(.0, 1.0), event_cum_pmf(3), cluster_cum_pmf() {}


    uint draw_event_index() {
        return draw_index(event_cum_pmf);
    }


    uint draw_cluster_index() {
        return draw_index(*cluster_cum_pmf);
    }


    void update_event_cum_pmf(const std::vector<float>& event_rates) {
        update_cum_pmf(event_rates, &event_cum_pmf);
    }


    void update_cluster_cum_pmf(const std::vector<uint>& clusters, uint cluster_end) {
        update_cum_pmf(clusters, cluster_cum_pmf, cluster_end);
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

        // Initialize cluster_cum_pmf to match to the requested population size
        std::vector<float> cluster_cum_pmf(N);
        this->cluster_cum_pmf = & cluster_cum_pmf;

        // Set the event rates
        std::vector<float> rates {alpha, beta, theta};
        update_event_cum_pmf(rates);

        // Set the initial clusters
        std::vector<uint> clusters(N, 0);
        uint pop_size = 1;
        clusters[0] = 1;
        // Keep track of the last cluster index to optimize the cum_pmf computation
        uint cluster_end = 1;
        update_cluster_cum_pmf(clusters, cluster_end);

        while (pop_size < N && pop_size > 0) {
            // Draw the event
            uint event = draw_event_index();
            uint cluster = draw_cluster_index();

            if (event==0) {
                clusters[cluster] += 1;
                pop_size += 1;
                update_cluster_cum_pmf(clusters, cluster_end);
            }
            else if (event==1) {
                clusters[cluster] -= 1;
                pop_size -= 1;
                update_cluster_cum_pmf(clusters, cluster_end);
            }
            else if (event==2) {
                // If there is only one member in this cluster, we do not need to do anything
                if (clusters[cluster] > 1) {
                    clusters[cluster] -= 1;

                    // Find an empty place and begin a new cluster
                    for (auto i = clusters.begin(); i != clusters.end(); ++i) {
                        if (*i == 0) {
                            *i = 1;
                            cluster_end = std::max(cluster_end, (uint)(i-clusters.begin()) + 1);
                            break;
                        }
                    }

                    update_cluster_cum_pmf(clusters, cluster_end);
                }
            }
        }

        this->cluster_cum_pmf = NULL;
        return clusters;
    }

};



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

    if (argc < 5) {
        // Inform the user how to use the program
        std::cout << "Usage is: bdm <alpha> <delta> <theta> <N> [--seed <seed>]\n";
        return 0;
    }

    float alpha = std::strtof(argv[1], NULL);
    float delta = std::strtof(argv[2], NULL);
    float theta = std::strtof(argv[3], NULL);
    uint N = (uint) std::stoul(argv[4], NULL);

    uint seed = (uint) time(0);
    if (argc == 7 and strcmp(argv[5], "--seed") == 0)
        seed = (uint) std::stoul(argv[6]);

    // Construct the simulator and simulate a population
    BDM bdm(seed);
    std::vector<uint> pop = bdm.simulate_population(alpha, delta, theta, N);

    // Write to stdout
    for (auto i = pop.begin(); i != pop.end(); ++i) std::cout << *i << ' ';

    return 0;
}
