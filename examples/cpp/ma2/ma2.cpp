#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <iterator>

/*
The 2nd order Moving Average model (MA2)
y_i = u_i + t_1 * u_{i-1} + t_2 * u_{i-2}
with command line arguments t_1, t_2 and white noise u_i.
*/

int main(int argc, char* argv[]) {

    if (argc < 3) {
        std::cerr << "Please give 2 coefficients as arguments, e.g. ma2.exe 0.6 0.2" << std::endl;
        std::cerr << "Random seed (unsigned int) can be given as the third argument." << std::endl;
        std::cerr << "Length of output (int) can be given as the fourth argument (default 100)." << std::endl;
        return 1;
    }

    // parse parameters
    double t1 = std::strtod(argv[1], NULL);
    double t2 = std::strtod(argv[2], NULL);

    // parse random seed, if given
    unsigned int seed;
    if (argc > 3) {
        seed = std::stoi(argv[3], NULL);
    } else {
        seed = (unsigned int) time(0);
    }

    // parse output length, if given
    int n;
    if (argc > 4) {
        n = std::stoi(argv[4], NULL);
    } else {
        n = 100;
    }

    // initialize Gaussian pseudo-random number generator
    std::mt19937 gen(seed);
    std::normal_distribution<> dist(0., 1.);

    // generate n+2 samples from N(0,1)
    std::vector<double> u (n+2, 0.);
    for (int ii=0; ii < n+2; ii++) {
        u[ii] = dist(gen);
    }

    // generate MA(2) dataset
    std::vector<double> ma2 (n, 0.);
    for (int ii=0; ii < n; ii++) {
        ma2[ii] = u[ii+2] + t1 * u[ii+1] + t2 * u[ii];
    }

    // print output to stdout
    std::copy(ma2.begin(), ma2.end(), std::ostream_iterator<double>(std::cout, " "));

    return 0;
}
