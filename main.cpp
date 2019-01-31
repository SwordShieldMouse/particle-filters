//
//  main.cpp
//  branching
//
//  Created by Alan on 2017-12-21.
//  Copyright © 2017 Alan. All rights reserved.
//
#include <iostream>
#include <thread>
#include <future>
#include <random>
#include <chrono>
#include <cmath>
#include <array>
#include <fstream>
#include <vector>
#include <algorithm>
#include "functions.h"
#include <string>
#include <map>

using namespace std;


int main()
{

    ofstream output;
    output.open(filename);
    
    //vector<thread> threads;
    vector< future< vector<vector<string>> > > futures;

    map<string, Particle_Filter*> filters;
    
    filters["combined branching"] = new Combined_Branching();
    filters["ablated three-chain"] = new Ablated_Three_Chain();
    filters["ablated two-chain"] = new Ablated_Two_Chain();
    filters["ablated antithetic two-chain"] = new Ablated_Antithetic_Two_Chain();
    filters["interacting combined"] = new Interacting_Combined();
    filters["minimum variance"] = new Minimum_Variance();
    filters["bootstrap"] = new Bootstrap();


    random_device rd;
    mt19937_64 generator(rd());
    
    //uniform_real_distribution<double> uniform_distribution(0.0, 1.0);

    vector<double> R;//, C, Q, C_EFF, C_NONEFF;

    initialize_R(R);
    //initialize_dyn_branch_param(C, Q);
    //initialize_eff_branch_param(C_EFF, C_NONEFF);

    //bool skip_parameter = false;
    
    // generate signal-observation pairs
    vector<array<double, T>> xs(trials);
    vector<array<double, T>> ys(trials);
    for (int i = 0; i < trials; i++) {
        array<array<double, T>, 2> signal_obs = Classic_Model::generate_signal_obs(generator);
        xs[i] = signal_obs[0];
        ys[i] = signal_obs[1];
    }
    
    // start the threads, one for each filter
    // allows for possibility of multithreading when time doesn't matter (e.g., variation experiments)
    for (auto const& [name, filter] : filters) {
        futures.push_back(async(launch::deferred), run_experiment, R, xs, ys, name, filter); // Use for performance experiments where we don't want to multithread
        //futures.push_back(async(launch::async, run_experiment, R, xs, ys, name, filter)); // use if multithreading is ok and won't impact individual performance
    }
    
    // get the values from the threads
    vector< vector<vector<string>> > experiment_data;
    for (int i = 0; i < futures.size(); i++) {
        experiment_data.push_back(futures[i].get());
    }
    
    // write all the data to file
    for (auto filter : experiment_data) {
        for (auto data : filter) {
            write_file(data, output);
        }
    }


    /*for (auto const& [name, func] : filters) {
        cout << "starting " << name << endl;
        for (double r : R) {
            // For particle variation experiments
            //if (name == "combined branching") {
            //    if (r != )
            //}
            
            skip_parameter = false;

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

                    clock_t tStart = clock();
                    
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


                    for (int n = 1; n < T; n++)
                    {
                        // for updating offspring number
                        int particle_number_update = 0;
                        
                        model.update_weights(N, n, particle_number);
                        model.evolve(particle_number, n, generator);


                        // Resampling
                        array<double, 2> percents = func->resample(particle_number, particle_number_update, model.get_avg_weight(), a, b, model.get_particles(), uniform_distribution, generator);

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

                        if (particle_number >= max_N) {
                            cout << "exceeded particle limit" << endl;
                            skip_parameter = true;
                            break;
                        }


                    }
                    if (skip_parameter == true) {
                        break;
                    }
                    times.push_back((long double) (clock() - tStart) / CLOCKS_PER_SEC);
                    
                    avg_res += model.get_final_residual();

                    variances.push_back(variance(particle_numbers));
                    avg_N += average(particle_numbers);
                    avg_min += curr_min;
                    avg_max += curr_max;
                }
                if (skip_parameter == true) {
                    continue;
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
                write_file(data, output);

                // CHANGE THIS when changing model
                if (avg_res <= classic_error) {
                    break;
                }
            }
        }
    }*/

    cout << "finished writing" << endl;

    output.close();

    return 0;
}
