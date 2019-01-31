//
//  constants.h
//  branching
//
//  Created by Alan on 2018-05-11.
//  Copyright © 2018 Alan. All rights reserved.
//

#ifndef constants_h
#define constants_h

#include <vector>

//const int max_N = 10000;
const int max_N = 20000;
const int T = 1000;
const int N_start = 150;
const int N_increment = 10;
const std::string filename = "classic-vari-r=1,6-T=1000.csv";
std::vector<double> N_list = {1000, 10000};

const int trials = 1000;
const double upper_L = pow(10, 10);
const double lower_L = pow(10, -10);

const double test_error = 5.5;
const double rom_error = 25.0;
const double classic_error = 14;
const double sv_error = 10;


#endif /* constants_h */
