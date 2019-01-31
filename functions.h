//
//  functions.h
//  field-branching
//
//  Created by Alan on 2018-05-11.
//  Copyright © 2018 Alan. All rights reserved.
//

#ifndef functions_h
#define functions_h

#include <vector>
#include <array>
#include <random>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <string>
#include <cmath>
#include "constants.h"

using namespace std;

double frac_part(double a) {
    // gives the fractional part of a real number
    return (a - int(a));
}

/* define a bounded function f */
double f(double x)
{
    double y;
    if (x > 30) {
        y = 30;
    } else if (x < -30) {
        y = -30;
    } else {
        y = x;
    }
    return y;
}

double g(double x) {
    if (x > 1000) {
        return 1000;
    } else if (x < -1000) {
        return -1000;
    } else {
        return x;
    }
}

double variance(vector<double> data) {
    int N = data.size();
    if (N == 0) {
        cout << "no particles" << endl;
        return 0;
    }
    double squared = 0.0;
    double mean = 0.0;
    for (double i : data) {
        squared += i * i;
        mean += i;
    }
    squared /= N;
    mean /= N;
    mean *= mean;
    return N / (N - 1) * (squared - mean);
}

double distance(double x, double y) {
    return sqrt(x * x + y * y);
}

double average(vector<double> data) {
    double sum = 0;
    for (auto i : data) {
        sum += i;
    }
    return sum / data.size();
}

class Model {
protected:
    double residual;
    double avg_weight;
    double prev_avg_weight;
    
    virtual double weight(double x, double y) = 0;
    
public:
    Model () {}
    double get_avg_weight() {
        return avg_weight;
    }
    virtual double get_final_residual() = 0;
    
    // return the addresses of particles for use in resampling
    virtual vector<array<double, max_N>*> get_particles() = 0;
};

class Test_Model : public Model {
protected:
    array<double, T> x, y, estimation;
    array<double, max_N> X, L;
    cauchy_distribution<double> cauchy_distribution;
    double signal_f, signal_1;
    
    double weight(double x, double y) {
        return (1 + y * y)/(1 + (y - x) * (y - x));
    }
    
public:
    Test_Model(int N, mt19937_64& generator) {
        /* simulate data of signal and observation */
        avg_weight = 0;
        prev_avg_weight = 0;
        residual = 0;
        signal_f = 0;
        signal_1 = 0;
        
        x[0] = cauchy_distribution(generator);
        y[0] = 0;
        for (int n = 1; n < T; n++)
        {
            x[n] = 0.95*x[n - 1] + 0.3*cauchy_distribution(generator);
            y[n] = x[n - 1] + cauchy_distribution(generator);
        }
        
        /* initialize particle, weight and  offspring-number */
        for (int k = 0; k < N; k++)
        {
            X[k] = cauchy_distribution(generator);
            L[k] = 1.0;
        }
    }
    
    int update_weights(int N, int n, int particle_number) {
        int factors = 0;
        double proportion = 1.0;
        avg_weight = 0;
        if (prev_avg_weight >= upper_L) {
            proportion = lower_L;
            factors++;
        } else if (prev_avg_weight <= lower_L) {
            proportion = upper_L;
            factors--;
        }
        for (int k = 0; k < particle_number; k++)
        {
            L[k] *= weight(X[k], y[n]) * proportion;
            avg_weight += L[k];
            //weight_square += L[k] * L[k];
        }
        avg_weight /= N;
        prev_avg_weight = avg_weight;
        
        return factors;
    }
    
    void evolve(int particle_number, int n, mt19937_64& generator) {
        signal_f = 0;
        signal_1 = 0;
        /* evolve particles and estimate the signal */
        for (int k = 0; k < particle_number; k++)
        {
            X[k] = 0.95*X[k] + 0.3*cauchy_distribution(generator);
            signal_f += L[k] * f(X[k]);
            signal_1 += L[k];
        }
        estimation[n] = signal_f / signal_1;
        
        // Update the residual value
        residual += pow(estimation[n] - f(x[n]), 2);
    }
    
    vector<array<double, max_N>*> get_particles() {
        return {&X, &L};
    }
    
    double get_final_residual() {
        return sqrt(residual / T);
    }
};

class Classic_Model : public Model {
protected:
    array<double, T> x, y, estimation;
    array<double, max_N> X, L;
    double signal_f, signal_1;
    normal_distribution<double> u_dist;
    cauchy_distribution<double> v_dist;
    
    double weight(double x, double y) {
        /*double numerator = exp(-(y - h(x)) * (y - h(x)) / 2);
        double denominator = exp(-y * y / 2);
        return numerator / denominator;*/
        //return exp(-(y - h(x)) * (y - h(x)) / 2 + y * y / 2);
        return (1 + y * y)/(1 + (y - h(x)) * (y - h(x)));
    }
             
    // part of the observation eqn
    double h(double x) {
        return x * x / 20;
    }
    
    
public:
    Classic_Model(int N, mt19937_64& generator) {
        u_dist = normal_distribution<double>(0, 10);
        v_dist = cauchy_distribution<double>(0, 1);
        
        /* simulate data of signal and observation */
        avg_weight = 0;
        prev_avg_weight = 0;
        residual = 0;
        signal_f = 0;
        signal_1 = 0;

        /*x[0] = u_dist(generator);
        y[0] = v_dist(generator);
        for (int n = 1; n < T; n++)
        {
            x[n] = x[n - 1] / 2 + 25 * x[n - 1] / (1 + x[n-1] * x[n - 1]) + 8 * cos(1.2 * (n - 1)) + u_dist(generator);
            y[n] = x[n] * x[n] / 20 + v_dist(generator);
        }*/
        
        /* initialize particle, weight and  offspring-number */
        for (int k = 0; k < N; k++)
        {
            X[k] = u_dist(generator);
            L[k] = 1.0;
        }
    }
    
    static array<array<double, T>, 2> generate_signal_obs(mt19937_64& generator) {
        array<array<double, T>, 2> signal_obs;
        normal_distribution<double> u(0, 10);
        cauchy_distribution<double> v(0, 1);
        array<double, T> temp_x;
        temp_x[0] = u(generator);
        array<double, T> temp_y;
        temp_y[0] = v(generator);
        for (int n = 1; n < T; n++)
        {
            temp_x[n] = temp_x[n - 1] / 2 + 25 * temp_x[n - 1] / (1 + temp_x[n-1] * temp_x[n - 1]) + 8 * cos(1.2 * (n - 1)) + u(generator);
            temp_y[n] = temp_x[n] * temp_x[n] / 20 + v(generator);
        }
        signal_obs[0] = temp_x;
        signal_obs[1] = temp_y;
        return signal_obs;
    }
    
    void set_signal_obs(array<double, T> x, array<double, T> y) {
        this->x = x;
        this->y = y;
    }
    
    int update_weights(int N, int n, int particle_number) {
        int factors = 0;
        double proportion = 1.0;
        avg_weight = 0;
        if (prev_avg_weight >= upper_L) {
            proportion = lower_L;
            factors++;
        } else if (prev_avg_weight <= lower_L && prev_avg_weight < 1) {
            proportion = upper_L;
            factors--;
        }
        for (int k = 0; k < particle_number; k++)
        {
            L[k] *= weight(X[k], y[n]) * proportion;
            avg_weight += L[k];
        }
        avg_weight /= N;
        prev_avg_weight = avg_weight;
        
        return factors;
    }
    
    void evolve(int particle_number, int n, mt19937_64& generator) {
        signal_f = 0;
        signal_1 = 0;
        /* evolve particles and estimate the signal */
        for (int k = 0; k < particle_number; k++)
        {
            X[k] = X[k] / 2 + 25 * X[k] / (1 + X[k] * X[k]) + 8 * cos(1.2 * (n - 1)) + u_dist(generator);
            signal_f += L[k] * g(X[k]);
            signal_1 += L[k];
        }
        estimation[n] = signal_f / signal_1;
        
        // Update the residual value
        residual += pow(estimation[n] - g(x[n]), 2);
    }
    
    vector<array<double, max_N>*> get_particles() {
        return {&X, &L};
    }
    
    double get_final_residual() {
        return sqrt(residual / T);
    }
};

class Stochastic_Volatility_Model : public Model {
protected:
    array<double, T> x, y, estimation;
    array<double, max_N> X, L;
    double signal_f, signal_1;
    normal_distribution<double> u_dist;
    student_t_distribution<double> v_dist;
    
    double weight(double x, double y) {
        double a = y - h(x);
        double numerator = pow(1 + a * a / 10, -(10 + 1) / 2.0);
        double denominator = pow(1 + y * y / 10, -(10 + 1) / 2.0);
        return numerator / denominator;
    }
    
    // part of the observation eqn
    double h(double x) {
        return exp(x / 2);
    }
    
    // part of the evolution equation
    double l(double x) {
        return 0.5 + 0.95 * x;
    }
    
public:
    Stochastic_Volatility_Model(int N, mt19937_64& generator) {
        u_dist = normal_distribution<double>(0, 10);
        v_dist = student_t_distribution<double>(10);
        
        /* simulate data of signal and observation */
        avg_weight = 0;
        prev_avg_weight = 0;
        residual = 0;
        signal_f = 0;
        signal_1 = 0;
        
        x[0] = u_dist(generator);
        y[0] = 0;
        for (int n = 1; n < T; n++)
        {
            x[n] = l(x[n - 1]) + u_dist(generator);
            y[n] = h(x[n - 1]) + v_dist(generator);
        }
        
        /* initialize particle, weight and  offspring-number */
        for (int k = 0; k < N; k++)
        {
            X[k] = u_dist(generator);
            L[k] = 1.0;
        }
    }
    
    int update_weights(int N, int n, int particle_number) {
        int factors = 0;
        double proportion = 1.0;
        avg_weight = 0;
        /*if (prev_avg_weight >= upper_L) {
            proportion = lower_L;
            factors++;
        } else if (prev_avg_weight <= lower_L && prev_avg_weight < 1) {
            proportion = upper_L;
            factors--;
        }*/
        for (int k = 0; k < particle_number; k++)
        {
            L[k] *= weight(X[k], y[n]);
            avg_weight += L[k];
        }
        avg_weight /= N;
        for (int k = 0; k < particle_number; k++)
        {
            L[k] /= avg_weight;
        }
        // make sure weights don't get too small or too large
        if (avg_weight == 0) {
            cin.get();
        }
        prev_avg_weight = avg_weight;
        
        return factors;
    }
    
    void evolve(int particle_number, int n, mt19937_64& generator) {
        signal_f = 0;
        signal_1 = 0;
        /* evolve particles and estimate the signal */
        for (int k = 0; k < particle_number; k++)
        {
            X[k] = l(X[k - 1]) + u_dist(generator);
            signal_f += L[k] * f(X[k]);
            signal_1 += L[k];
        }
        estimation[n] = signal_f / signal_1;
        
        // Update the residual value
        residual += pow(estimation[n] - f(x[n]), 2);
    }
    
    vector<array<double, max_N>*> get_particles() {
        return {&X, &L};
    }
    
    double get_final_residual() {
        return sqrt(residual / T);
    }
};


class Range_Only_Model : public Model {
protected:
    array<double, T> x, y, v_x, v_y, d, estimation_X, estimation_Y;
    array<double, max_N> X, Y, V_x, V_y, L;
    double D, unnormalized_filter, f_x, f_y;
    
    double weight(double x, double y) {
        return (1 +  100 * y * y)/(1 +  100 * (y - x) * (y - x));
    }
    
public:
    Range_Only_Model (int N, cauchy_distribution<double>& cauchy_distribution, normal_distribution<double>& normal_distribution, mt19937_64& generator) {
        avg_weight = 0;
        prev_avg_weight = 0;
        residual = 0;
        D = 0;
        unnormalized_filter = 0;
        f_x = 0;
        f_y = 0;
        
        x[0] = 10 * cauchy_distribution(generator);
        y[0] = 10 * cauchy_distribution(generator);
        v_x[0] = 5 * normal_distribution(generator);
        v_y[0] = 5 * normal_distribution(generator);
        
        /* simulate data of signal and observation */
        for (int n = 1; n < T; n++)
        {
            x[n] = 0.5*x[n - 1] + v_x[n - 1] + cauchy_distribution(generator);
            y[n] = 0.5*y[n - 1] + v_y[n - 1] + cauchy_distribution(generator);
            v_x[n] = 0.95*v_x[n - 1] + normal_distribution(generator);
            v_y[n] = 0.95*v_y[n - 1] + normal_distribution(generator);
            d[n] = distance(x[n - 1], y[n - 1]) + 0.1*cauchy_distribution(generator);
        }
        
        /* initialize particle, weight and  offspring-number */
        for (int k = 0; k < N; k++)
        {
            X[k] = 10 * cauchy_distribution(generator);
            Y[k] = 10 * cauchy_distribution(generator);
            V_x[k] = 5 * normal_distribution(generator);  // cauchy_distribution(generator);
            V_y[k] = 5 * normal_distribution(generator);  // cauchy_distribution(generator);
            L[k] = 1;
        }
    }
    
    int update_weights(int N, int n, int particle_number) {
        int factors = 0;
        unnormalized_filter = 0;
        double proportion = 1.0;
        if (prev_avg_weight >= upper_L) {
            proportion = lower_L;
            factors++;
        } else if (prev_avg_weight <= lower_L) {
            proportion = upper_L;
            factors--;
        }
        /* weight each particle by observation and calculate average weight */
        for (int k = 0; k < particle_number; k++)
        {
            D = distance(X[k], Y[k]);
            L[k] *= weight(D, d[n]) * proportion;
            unnormalized_filter += L[k];
        }
        avg_weight = unnormalized_filter / N;
        prev_avg_weight = avg_weight;
        
        return factors;
    }
    
    void evolve(int particle_number, int n, cauchy_distribution<double>& cauchy_distribution, normal_distribution<double>& normal_distribution, mt19937_64& generator) {
        f_x = 0;
        f_y = 0;
        for (int k = 0; k < particle_number; k++) {
            /* x[n] = x[n - 1] + v_x[n - 1] + 0.3*cauchy_distribution(generator); */
            /* v_x[n] = v_x[n - 1] + normal_distribution(generator) + cauchy_distribution(generator); */
            X[k] = 0.5*X[k] + V_x[k] + cauchy_distribution(generator);
            Y[k] = 0.5*Y[k] + V_y[k] + cauchy_distribution(generator);
            V_x[k] = 0.95*V_x[k] + normal_distribution(generator); // +0.3*cauchy_distribution(generator);
            V_y[k] = 0.95*V_y[k] + normal_distribution(generator); // +0.3*cauchy_distribution(generator);
            f_x += L[k] * g(X[k]);
            f_y += L[k] * g(Y[k]);
            // f_v_x = f_v_x + L[k] * v_x[k];
            // f_v_y = f_v_y + L[k] * v_y[k];
        }
        
        estimation_X[n] = f_x / unnormalized_filter;
        estimation_Y[n] = f_y / unnormalized_filter;
        
        residual += sqrt(pow(estimation_X[n] - g(x[n]), 2) + pow(estimation_Y[n] - g(y[n]), 2));
    }
    
    vector<array<double, max_N>*> get_particles() {
        return {&X, &Y, &V_x, &V_y, &L};
    }
    
    double get_final_residual() {
        return residual / T;
    }
};

class Particle_Filter {
protected:
    int resample_particle_number;
    int nonresample_particle_number;
    vector<array<double, max_N>> updates;
    vector<array<double, max_N>> resamples;
    
    Particle_Filter () {}
    
    // Segregate particles for partial resampling
    void segregate(double a, double b, double avg_weight, int particle_number, vector<array<double, max_N>*> originals) {
        int originals_length = originals.size();
        for (int k = 0; k < particle_number; k++)
        {
            double L_k = (*originals[originals_length - 1])[k];
            if (L_k <= a * avg_weight || L_k >= b * avg_weight) {
                for (int j = 0; j < originals_length; j++) {
                    resamples[j][resample_particle_number] = (*originals[j])[k];
                }
                resample_particle_number++;
            } else {
                for (int j = 0; j < originals_length; j++) {
                    updates[j][nonresample_particle_number] = (*originals[j])[k];
                }
                nonresample_particle_number++;
            }
        }
    }
    
    // For updating particles because of offspring
    void update_offspring(int begin, int end, int update_length, int particle_index, double avg_weight) {
        for (int i = begin; i < end; i++) {
            for (int j = 0; j < update_length - 1; j++) {
                updates[j][i] = resamples[j][particle_index];
            }
            updates[update_length - 1][i] = avg_weight;
        }
    }
    
    // For returning the final update
    void get_final_update(vector<array<double, max_N>*> originals) {
        int update_length = originals.size();
        for (int j = 0; j < update_length; j++) {
            (*originals[j]) = updates[j];
        }
    }
    
    virtual void resample(int& particle_number_update, double avg_weight, uniform_real_distribution<double>& uniform_distribution, mt19937_64& generator, int update_length) = 0;
    
public:
    // resample and return two statistics: 1. % of particles negatively correlated as percentage of particles resampled. 2. % of particles resampled.
    array<double, 2> resample(int particle_number, int& particle_number_update, double avg_weight, double a, double b, vector<array<double, max_N>*> originals,  uniform_real_distribution<double>& uniform_distribution, mt19937_64& generator) {
        
        // Prepare for resampling
        resample_particle_number = 0;
        nonresample_particle_number = 0;
        
        int update_length = originals.size();
        
        resamples.resize(update_length);
        updates.resize(update_length);
        
        Particle_Filter::segregate(a, b, avg_weight, particle_number, originals);
        
        particle_number_update = nonresample_particle_number;
        
        // The actual, unique resampling method
        resample(particle_number_update, avg_weight, uniform_distribution, generator, update_length);
        
        // Final particle update
        Particle_Filter::get_final_update(originals);
        
        // 1st item returned is neg corr. % of resampled, 2nd item is % resampled
        array<double, 2> results;
        results[0] = 1.0;
        results[1] = (double) resample_particle_number / particle_number;
        return results;
    }
};

class Stratified_Two_Chain : public Particle_Filter {
public:
    void resample(int& particle_number_update, double avg_weight,  uniform_real_distribution<double>& uniform_distribution, mt19937_64& generator, int update_length) {
        // generate the uniforms and implement a shuffle
        double u [resample_particle_number];
        
        for (int k = 0; k < resample_particle_number; k++) {
            u[k] = (k + uniform_distribution(generator)) / resample_particle_number;
        }
        
        // fischer shuffle
        /*for (int i = resample_particle_number; i >= 1; i--) {
         uniform_int_distribution<int> uni_int (0, i);
         int j = uni_int(generator);
         double temp = u[i];
         u[i] = u[j];
         u[j] = temp;
         }*/
        
        int resample_counter = 0;
        int curr_value = 0;
        double curr_prob = 0.0;
        
        // keep track of neg. corr. particles
        int corr_particles = 0;
        
        int prev_v1 = 0;
        double prev_pmf1 = 0.0;
        double prev_p1 = 0.0;
        
        for (int k = 0; k < resample_particle_number; k++) {
            double L_A = resamples[update_length - 1][k] / avg_weight;
            double p = frac_part(L_A);
            
            if ((resample_counter + 1) % 2 == 0) {
                double cov = -sqrt(p * prev_p1 * (1.0 - p) * (1.0 - prev_p1));
                if (prev_v1 == 1) {
                    curr_prob = p + cov / prev_pmf1;
                } else {
                    curr_prob = p - cov / prev_pmf1;
                }
            } else {
                curr_prob = p;
            }
            
            // only update if probabilities will make sense
            if (curr_prob > 1 || curr_prob <= 0) {
                curr_prob = p;
            } else if (curr_prob != p) {
                corr_particles++;
            }
            
            
            if (u[k] <= curr_prob) {
                curr_value = 1;
            } else {
                curr_value = 0;
            }
            
            resample_counter++;
            prev_v1 = curr_value;
            prev_p1 = p;
            if (prev_v1 == 1) {
                prev_pmf1 = p;
            } else {
                prev_pmf1 = 1.0 - p;
            }
            
            //cout << curr_value << " " << curr_prob << endl;
            
            int offspring_number = int(L_A) + curr_value;
            
            /* drop particle and weight to "update container" */
            Particle_Filter::update_offspring(particle_number_update, particle_number_update + offspring_number, update_length, k, avg_weight);
            
            /* update particle number */
            particle_number_update += offspring_number;
        }
    }
};

class Ablated_Two_Chain : public Particle_Filter {
public:
    void resample(int& particle_number_update, double avg_weight,  uniform_real_distribution<double>& uniform_distribution, mt19937_64& generator, int update_length) {
        
        int curr_value = 0;
        double curr_prob = 0.0;
        
        // keep track of neg. corr. particles
        int corr_particles = 0;
        
        int prev_v1 = 0;
        double prev_pmf1 = 0.0;
        double prev_p1 = 0.0;
        
        for (int k = 0; k < resample_particle_number; k++) {
            double L_A = resamples[update_length - 1][k] / avg_weight;
            double p = frac_part(L_A);
            
            if ((k + 1) % 2 == 0) {
                double cov = -sqrt(p * prev_p1 * (1.0 - p) * (1.0 - prev_p1));
                if (prev_v1 == 1) {
                    curr_prob = p + cov / prev_pmf1;
                } else {
                    curr_prob = p - cov / prev_pmf1;
                }
            } else {
                curr_prob = p;
            }
            
            // only update if probabilities will make sense
            if (curr_prob > 1 || curr_prob <= 0) {
                curr_prob = p;
            } else if (curr_prob != p) {
                corr_particles++;
            }
            
            
            if (uniform_distribution(generator) <= curr_prob) {
                curr_value = 1;
            } else {
                curr_value = 0;
            }
            
            prev_v1 = curr_value;
            prev_p1 = p;
            if (prev_v1 == 1) {
                prev_pmf1 = p;
            } else {
                prev_pmf1 = 1.0 - p;
            }
            
            //cout << curr_value << " " << curr_prob << endl;
            
            
            int offspring_number = int(L_A) + curr_value;
            
            /* drop particle and weight to "update container" */
            Particle_Filter::update_offspring(particle_number_update, particle_number_update + offspring_number, update_length, k, avg_weight);
            
            /* update particle number */
            particle_number_update += offspring_number;
        }
    }
};

class Ablated_Antithetic_Two_Chain : public Particle_Filter {
public:
    void resample(int& particle_number_update, double avg_weight,  uniform_real_distribution<double>& uniform_distribution, mt19937_64& generator, int update_length) {
        
        int curr_value = 0;
        double curr_prob = 0.0;
        
        // keep track of neg. corr. particles
        int corr_particles = 0;
        
        int prev_v1 = 0;
        double prev_pmf1 = 0.0;
        double prev_p1 = 0.0;
        
        double u;
        
        for (int k = 0; k < resample_particle_number; k++) {
            double L_A = resamples[update_length - 1][k] / avg_weight;
            double p = frac_part(L_A);
            
            if ((k + 1) % 2 == 0) {
                double cov = -sqrt(p * prev_p1 * (1.0 - p) * (1.0 - prev_p1));
                if (prev_v1 == 1) {
                    curr_prob = p + cov / prev_pmf1;
                } else {
                    curr_prob = p - cov / prev_pmf1;
                }
                u = 1 - u;
            } else {
                curr_prob = p;
                u = uniform_distribution(generator);
            }
            
            // only update if probabilities will make sense
            if (curr_prob > 1 || curr_prob <= 0) {
                curr_prob = p;
            } else if (curr_prob != p) {
                corr_particles++;
            }
            
            
            if (u <= curr_prob) {
                curr_value = 1;
            } else {
                curr_value = 0;
            }
            
            prev_v1 = curr_value;
            prev_p1 = p;
            if (prev_v1 == 1) {
                prev_pmf1 = p;
            } else {
                prev_pmf1 = 1.0 - p;
            }
            
            //cout << curr_value << " " << curr_prob << endl;
            
            
            int offspring_number = int(L_A) + curr_value;
            
            /* drop particle and weight to "update container" */
            Particle_Filter::update_offspring(particle_number_update, particle_number_update + offspring_number, update_length, k, avg_weight);
            
            /* update particle number */
            particle_number_update += offspring_number;
        }
    }
};

class Ablated_Three_Chain : public Particle_Filter {
public:
    void resample(int& particle_number_update, double avg_weight,  uniform_real_distribution<double>& uniform_distribution, mt19937_64& generator, int update_length) {
        /* start resample */
        int curr_value = 0;
        double curr_prob = 0.0;
        
        
        // keep track of neg. corr. particles
        //int corr_particles = 0;
        //int corr_batch = 0;
        
        int prev_v1 = 0;
        int prev_v2 = 0; // the most recent value
        double prev_pmf1 = 0.0;
        double prev_pmf2 = 0.0; // the most recent pmf
        double prev_p1 = 0.0;
        double prev_p2 = 0.0; // the most recent probability
        
        // last element of array means most recent value
        //vector<double> probs(2);
        //vector<double> values(2);
        
        double total_pmf = 1.0;
        
        for (int k = 0; k < resample_particle_number; k++) {
            double L_A = resamples[update_length - 1][k] / avg_weight;
            double p = frac_part(L_A);
            
            //corr_batch++;
            
            double sum = 0;
            
            if ((k + 1) % 3 == 2) { // for second particle
                double cov = -sqrt(p * prev_p2 * (1.0 - p) * (1.0 - prev_p2) / 4.0);
                if (prev_v2 == 1) {
                    curr_prob = p + cov / total_pmf;
                } else {
                    curr_prob = p - cov / total_pmf;
                }
            } else if ((k + 1) % 3 == 0) { // for third particle
                double s_31 = -sqrt(prev_p1 * (1.0 - prev_p1) * p * (1.0 - p) / 4.0);
                double s_32 = -sqrt(prev_p2 * (1.0 - prev_p2) * p * (1.0 - p) / 4.0);
                double g1 = (prev_v1 - prev_p1) / (prev_p1 * (1.0 - prev_p1));
                double g2 = (prev_v2 - prev_p2) / (prev_p2 * (1.0 - prev_p2));
                double h = prev_pmf1 * prev_pmf2;
                sum = h * (g1 * s_31 + g2 * s_32);
                curr_prob = p + sum / total_pmf;
            } else { // for first particle
                curr_prob = p;
                total_pmf = 1.0;
            }
            
            //cout << p << " " << cov << " " << gh << " " << curr_prob << endl;
            
            // only update if probabilities will make sense
            if (curr_prob > 1 || curr_prob <= 0) {
                curr_prob = p;
            }
            
            
            if (uniform_distribution(generator) <= curr_prob) {
                curr_value = 1;
                total_pmf *= curr_prob;
            } else {
                curr_value = 0;
                total_pmf *= 1.0 - curr_prob;
            }
            
            prev_v1 = prev_v2;
            prev_p1 = prev_p2;
            prev_pmf1 = prev_pmf2;
            prev_v2 = curr_value;
            prev_p2 = p;
            if (prev_v2 == 1) {
                prev_pmf2 = p;
            } else {
                prev_pmf2 = 1.0 - p;
            }
            
            int offspring_number = int(L_A) + curr_value;
            
            /* drop particle and weight to "update container" */
            Particle_Filter::update_offspring(particle_number_update, particle_number_update + offspring_number, update_length, k, avg_weight);
            
            /* update particle number */
            particle_number_update += offspring_number;
        }
    }
};

class Stratified_Three_Chain : public Particle_Filter {
public:
    void resample(int& particle_number_update, double avg_weight,  uniform_real_distribution<double>& uniform_distribution, mt19937_64& generator, int update_length) {
        // generate the uniforms and implement a shuffle
        double u [resample_particle_number];
        
        for (int k = 0; k < resample_particle_number; k++) {
            u[k] = (k + uniform_distribution(generator)) / resample_particle_number;
        }
        
        // fischer shuffle
        /*for (int i = resample_particle_number; i >= 1; i--) {
         uniform_int_distribution<int> uni_int (0, i);
         int j = uni_int(generator);
         double temp = u[i];
         u[i] = u[j];
         u[j] = temp;
         }*/
        
        /* start resample */
        int curr_value = 0;
        double curr_prob = 0.0;
        
        
        // keep track of neg. corr. particles
        int corr_particles = 0;
        int corr_batch = 0;
        
        int prev_v1 = 0;
        int prev_v2 = 0; // the most recent value
        double prev_pmf1 = 0.0;
        double prev_pmf2 = 0.0; // the most recent pmf
        double prev_p1 = 0.0;
        double prev_p2 = 0.0; // the most recent probability
        
        // last element of array means most recent value
        //vector<double> probs(2);
        //vector<double> values(2);
        
        double total_pmf = 1.0;
        
        for (int k = 0; k < resample_particle_number; k++) {
            double L_A = resamples[update_length - 1][k] / avg_weight;
            double p = frac_part(L_A);
            
            corr_batch++;
            
            double sum = 0;
            
            if ((k + 1) % 3 == 2) { // for second particle
                double cov = -sqrt(p * prev_p2 * (1.0 - p) * (1.0 - prev_p2) / 4.0);
                if (prev_v2 == 1) {
                    curr_prob = p + cov / total_pmf;
                } else {
                    curr_prob = p - cov / total_pmf;
                }
            } else if ((k + 1) % 3 == 0) { // for third particle
                double s_31 = -sqrt(prev_p1 * (1.0 - prev_p1) * p * (1.0 - p) / 4.0);
                double s_32 = -sqrt(prev_p2 * (1.0 - prev_p2) * p * (1.0 - p) / 4.0);
                double g1 = (prev_v1 - prev_p1) / (prev_p1 * (1.0 - prev_p1));
                double g2 = (prev_v2 - prev_p2) / (prev_p2 * (1.0 - prev_p2));
                double h = prev_pmf1 * prev_pmf2;
                sum = h * (g1 * s_31 + g2 * s_32);
                curr_prob = p + sum / total_pmf;
            } else { // for first particle
                curr_prob = p;
                total_pmf = 1.0;
            }
            
            //cout << p << " " << cov << " " << gh << " " << curr_prob << endl;
            
            // only update if probabilities will make sense
            if (curr_prob > 1 || curr_prob <= 0) {
                curr_prob = p;
            } else if (curr_prob != p) {
                corr_particles++;
            }
            
            
            if (u[k] <= curr_prob) {
                curr_value = 1;
                total_pmf *= curr_prob;
            } else {
                curr_value = 0;
                total_pmf *= 1.0 - curr_prob;
            }
            
            prev_v1 = prev_v2;
            prev_p1 = prev_p2;
            prev_pmf1 = prev_pmf2;
            prev_v2 = curr_value;
            prev_p2 = p;
            if (prev_v2 == 1) {
                prev_pmf2 = p;
            } else {
                prev_pmf2 = 1.0 - p;
            }
            
            int offspring_number = int(L_A) + curr_value;
            
            /* drop particle and weight to "update container" */
            Particle_Filter::update_offspring(particle_number_update, particle_number_update + offspring_number, update_length, k, avg_weight);
            
            /* update particle number */
            particle_number_update += offspring_number;
        }
    }
};

class Combined_Branching : public Particle_Filter {
public:
    void resample(int& particle_number_update, double avg_weight,  uniform_real_distribution<double>& uniform_distribution, mt19937_64& generator, int update_length) {
        
        // generate the uniforms and implement a shuffle
        double u [resample_particle_number];
        
        for (int k = 0; k < resample_particle_number; k++) {
            u[k] = (k + uniform_distribution(generator)) / resample_particle_number;
        }
        
        // fischer shuffle
        /*for (int i = resample_particle_number; i >= 1; i--) {
         uniform_int_distribution<int> uni_int (0, i);
         int j = uni_int(generator);
         double temp = u[i];
         u[i] = u[j];
         u[j] = temp;
         }*/
        
        for (int k = 0; k < resample_particle_number; k++)
        {
            double L_A = resamples[update_length - 1][k] / avg_weight;
            double p = frac_part(L_A);
            
            int rho = 0;
            if (u[k] <= p) {
                rho = 1;
            }
            
            int offspring = int(L_A) + rho;
            
            /* drop particle and weight to "update container" */
            
            Particle_Filter::update_offspring(particle_number_update, particle_number_update + offspring, update_length, k, avg_weight);
            
            /* update particle number */
            particle_number_update += offspring;
        }
    }
};

class Minimum_Variance : public Particle_Filter {
public:
    void resample(int& particle_number_update, double avg_weight,  uniform_real_distribution<double>& uniform_distribution, mt19937_64& generator, int update_length) {
        double new_avg_weight = 0;
        for (int i = 0; i < resample_particle_number; i++) {
            new_avg_weight += resamples[update_length - 1][i];
        }
        new_avg_weight /= resample_particle_number;
        
        /* start resample */
        int particle_bound = resample_particle_number;
        double W = resample_particle_number;
        
        
        for (int k = 0; k < resample_particle_number; k++) {
            // Legitimate just to divide by avg weight since we need to multiple by N in the original algorithm anyway
            // Divide by the new avg weight instead of the old to maintain particle control
            double weight = resamples[update_length - 1][k] / new_avg_weight;
            int children = floor(weight);
            double weight_frac = weight - children;
            double weight_diff = W - weight - int(W - weight);
            double W_frac = W - int(W);
            double r = uniform_distribution(generator);
            if (weight_frac + weight_diff < 1 && r * W_frac <= weight_frac){
                children += particle_bound - int(W);
            } else if (weight_frac + weight_diff >= 1 && r * (1.0 - W_frac) >= weight_frac - W_frac) {
                children += particle_bound - int(W);
            } else if (weight_frac + weight_diff >= 1 && r * (1.0 - W_frac) < weight_frac - W_frac) {
                children += 1;
            }
            
            // Update counters
            W -= weight;
            
            
            // Don't propogate more children than you have space for in particle_bound
            if (particle_bound - children <= 0) {
                children = particle_bound;
                particle_bound = 0;
            } else {
                particle_bound -= children;
            }
            
            
            /* drop particle and weight to "update container" */
            Particle_Filter::update_offspring(particle_number_update, particle_number_update + children, update_length, k, avg_weight);
            
            /* update particle number */
            particle_number_update += children;
            
            if (particle_bound <= 0 || W <= 0) {
                break;
            }
        }
    }
};

class Interacting_Combined : public Particle_Filter {
public:
    void resample(int& particle_number_update, double avg_weight,  uniform_real_distribution<double>& uniform_distribution, mt19937_64& generator, int update_length) {
        
        double new_total_weight = 0;
        for (int i = 0; i < resample_particle_number; i++) {
            new_total_weight += resamples[update_length - 1][i];
        }
        
        // number of particles we are preserving
        int S = 0;
        
        
        // normalize the weights
        vector<double> L(resample_particle_number);
        for (int i = 0; i < resample_particle_number; i++) {
            L[i] = resamples[update_length - 1][i] / new_total_weight;
        }
        
        
        // residual
        for (int j = 0; j < resample_particle_number; j++)
        {
            int k = 0;
            while (k < int(resample_particle_number * L[j]))
            {
                Particle_Filter::update_offspring(S + k + nonresample_particle_number, S+k + nonresample_particle_number + 1, update_length, j, avg_weight);
                k++;
            }
            S += k;
        }
        
        int R = resample_particle_number - S; // number of resampled particles
        vector<double> p(resample_particle_number + 1); // CDF for particles
        p[0] = 0;
        for (int i = 0; i < resample_particle_number; i++)
        {
            p[i + 1] = p[i] + frac_part(resample_particle_number * L[i]) / R;
        }
        
        int j = 1;
        for (int k = 0; k < R; k++)
        {
            double u = (uniform_distribution(generator) + k) / R;
            
            while (u >= p[j])
            {
                j++;
            }
            
            Particle_Filter::update_offspring(S+k + nonresample_particle_number, S+k + nonresample_particle_number + 1, update_length, j - 1, avg_weight);
        }
        
        // remember to update the particle number
        particle_number_update += resample_particle_number;
    }
};

class Bootstrap : public Particle_Filter {
public:
    void resample(int& particle_number_update, double avg_weight,  uniform_real_distribution<double>& uniform_distribution, mt19937_64& generator, int update_length) {
        
        double new_total_weight = 0;
        for (int i = 0; i < resample_particle_number; i++) {
            new_total_weight += resamples[update_length - 1][i];
        }
        
        // normalize the weights
        vector<double> L(resample_particle_number);
        for (int i = 0; i < resample_particle_number; i++) {
            L[i] = resamples[update_length - 1][i] / new_total_weight;
        }
        
        int j = resample_particle_number - 1;
        double v = 1;
        vector<double> p(resample_particle_number + 1);
        p[0] = 0;
        // create the cdf
        for (int i = 1; i <= resample_particle_number; i++) {
            p[i] = p[i-1] + L[i - 1];
        }
        
        for (double k = resample_particle_number; k > 0; k--) {
            double u = uniform_distribution(generator);
            v = pow(u, 1 / k)*v;
            
            while (v <= p[j])
            {
                j--;
            }
            
            Particle_Filter::update_offspring(k - 1 + nonresample_particle_number, k + nonresample_particle_number, update_length, j, avg_weight);
            
        }
        
        // remember to update the particle number
        particle_number_update += resample_particle_number;
    }
};


void initialize_R(vector<double>& R) {
    for (double i = 2.00; i <= 6.01; i += 0.05){
        R.push_back(i);
    }
}

void initialize_dyn_branch_param(vector<double>& C, vector<double>& Q) {
    for (double i = 1.7; i >= -1.0; i -= 0.1) {
        C.push_back(i);
        Q.push_back(i);
    }
}

void initialize_eff_branch_param(vector<double>& C_EFF, vector<double>& C_NONEFF) {
    for (int i = 32; i >= 1; i--) {
        C_EFF.push_back(i);
        C_NONEFF.push_back(i);
    }
}

void write_file(vector<string> values, ofstream& output) {
    bool first = true;
    for (string i : values) {
        if (first) {
            first = false;
        } else {
            output << ", ";
        }
        output << i;
    }
    output << "\n";
}

void print_data(vector<string> values) {
    bool first = true;
    for (string i : values) {
        if (first) {
            first = false;
        } else {
            cout << ", ";
        }
        cout << i;
    }
    cout << endl;
}

vector<string> vector_to_string(vector<double> data) {
    vector<string> a;
    for (auto i : data) {
        a.push_back(to_string(i));
    }
    return a;
}

vector<vector<string>> run_experiment(vector<double> R, vector<array<double, T>> xs, vector<array<double, T>> ys, string name, Particle_Filter* filter) {
    // run an experiment and return all the associated data, one line per run
    
    cout << "starting " << name << endl;
    
    random_device rd;
    mt19937_64 generator(rd());
    uniform_real_distribution<double> uniform_distribution(0, 1);
    
    vector<vector<string>> all_data;
    
    for (double r : R) {
        
        double a = 1.0 / r;
        double b = r;
        
        vector<double> times; // to hold the completion times so that we can calculate std. error later
        
        for (int N = N_start ;; N += N_increment) {
            double avg_res = 0.0;
            double avg_time = 0.0;
            
            vector<double> variances;
            
            double neg_corr_particles = 0.0;
            double resample_percent = 0.0;
            
            double min_particles = max_N;
            double max_particles = 0;
            double avg_N = 0.0;
            double avg_min = 0.0;
            double avg_max = 0.0;
            
            for (int trial = 0; trial < trials; trial++) {
                double curr_max = 0;
                double curr_min = max_N;
                
                vector<double> particle_numbers;
                
                chrono::time_point<chrono::high_resolution_clock> start, end;
                start = chrono::high_resolution_clock::now();
                //clock_t tStart = clock();
                
                // test model
                //Test_Model model(N, generator);
                
                // range-only model
                //Range_Only_Model model(N, cauchy_distribution, normal_distribution, generator);
                
                // modified classic model from li 2015
                Classic_Model model(N, generator);
                
                // ensure that all filters see same signal/obs for fair comparison
                model.set_signal_obs(xs[trial], ys[trial]);
                
                //Stochastic_Volatility_Model model(N, generator);
                
                int particle_number = N;
                
                
                /* start algorithm */
                for (int n = 1; n < T; n++)
                {
                    // for updating offspring number
                    int particle_number_update = 0;
                    
                    model.update_weights(N, n, particle_number);
                    model.evolve(particle_number, n, generator);
                    
                    
                    // Resampling
                    array<double, 2> percents = filter->resample(particle_number, particle_number_update, model.get_avg_weight(), a, b, model.get_particles(), uniform_distribution, generator);
                    
                    neg_corr_particles += percents[0];
                    resample_percent += percents[1];
                    
                    
                    particle_number = particle_number_update;
                    particle_numbers.push_back(particle_number);
                    
                    //cout << particle_number << endl;
                    if (particle_number >= max_particles) {
                        max_particles = particle_number;
                    }
                    if (particle_number <= min_particles) {
                        min_particles = particle_number;
                    }
                    if (particle_number >= curr_max) {
                        curr_max = particle_number;
                    }
                    if (particle_number <= curr_min) {
                        curr_min = particle_number;
                    }
                    
                }
                end = chrono::high_resolution_clock::now();
                auto diff = chrono::duration_cast<chrono::milliseconds> (end - start);
                times.push_back(diff.count());
                
                avg_res += model.get_final_residual();
                
                variances.push_back(variance(particle_numbers));
                avg_N += average(particle_numbers);
                avg_min += curr_min;
                avg_max += curr_max;
            }
            avg_res /= trials;
            avg_time = average(times);
            double std_time_error = sqrt(variance(times) / trials);
            neg_corr_particles /= trials * T;
            resample_percent /= trials * T;
            avg_N /= trials;
            avg_min /= trials;
            avg_max /= trials;
            
            double pooled_var = 0.0;
            for (double var : variances) {
                pooled_var += (T - 1) * var;
            }
            pooled_var /= (trials * T - trials);
            
            // variance in particle numbers
            //double sdev_percent = sqrt(pooled_var) / N; // as percent of set particles, but could be avg
            
            double sdev = sqrt(pooled_var);
            //double sdev_lower = avg_N - 2 * sdev;
            //double sdev_higher = avg_N + 2 * sdev;
            double sdev_range = 4 * sdev;
            double sdev_range_percent = sdev_range / avg_N;
            double range = max_particles - min_particles;
            double range_percent = range / avg_N;
            double avg_range = avg_max - avg_min;
            double avg_range_percent = avg_range / avg_N;
            
            
            vector<double> temp_data;
            
            // CHANGE THIS when changing models
            if (avg_res <= classic_error) {
                // 1.0 at the end means done
                temp_data = { r, (double)N, avg_res, avg_time, std_time_error, avg_time - 1.96 * std_time_error, avg_time + 1.96 * std_time_error, resample_percent, avg_N, avg_min, avg_max, avg_range_percent, sdev_range_percent, range_percent, 1.0 };
            } else {
                // 0.0 at the end means not done
                temp_data = { r, (double)N, avg_res, avg_time, std_time_error, avg_time - 1.96 * std_time_error, avg_time + 1.96 * std_time_error, resample_percent, avg_N, avg_min, avg_max, avg_range_percent, sdev_range_percent, range_percent, 0.0 };
            }
            
            vector<string> data = vector_to_string(temp_data);
            
            data.push_back(name);
            
            print_data(data);
            //write_file(data, output);
            all_data.push_back(data);
            
            // CHANGE THIS when changing model
            if (avg_res <= classic_error) {
                break;
            }
        }
    }
    cout << "finished " << name << endl;
    return all_data;
}


#endif /* functions_h */
