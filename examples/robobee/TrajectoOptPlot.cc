#pragma once

#include <string.h>
#include <iostream>
#include <fstream>

#include "drake/systems/trajectory_optimization/direct_collocation.h"

namespace drake {
namespace examples {
namespace robobee {

typedef trajectories::PiecewisePolynomial<double> PiecewisePolynomialType;

int TrajectoOptPlot(systems::trajectory_optimization::DirectCollocation dircol, std::string directory);
/// The Quadrotor - an underactuated aerial vehicle. This version of the
/// Quadrotor is implemented to match the dynamics of the plant specified in
/// the `quadrotor.urdf` model file.
//-[4] Plotting the trajopt result.

int TrajectoOptPlot(systems::trajectory_optimization::DirectCollocation dircol, std::string directory){
  
  std::string file_name_state ("state_trajopt.txt");

  std::string file_name_input ("input_trajopt.txt");
  std::string file_name_time_col_trajopt ("time_col_trajopt.txt");
  std::string file_name_state_col_trajopt ("state_col_trajopt.txt");
  std::string file_name_input_col_trajopt ("input_col_trajopt.txt");
  
  file_name_state = directory + file_name_state;
  file_name_time_col_trajopt = directory + file_name_time_col_trajopt;
  file_name_state = directory + file_name_state;
  file_name_state_col_trajopt = directory + file_name_state_col_trajopt;
  std::cout << file_name_input_col_trajopt;


//   std::ofstream output_file;
//   output_file.open("/home/patrick/Research/drake/examples/robobee/state_trajopt.txt");
//   if (!output_file.is_open()) {
//       std::cerr << "Problem opening solution output file.\n";
//   }
//   double N =200;
//   double T =pp_xtraj.end_time();
//   double tt;
//   Eigen::VectorXd x_opt;
//   for (int i = 0; i <= N; i++) {
//     tt = i/N * T;
//     x_opt = pp_xtraj.value(tt);
//     output_file << tt << '\t';    
    
//     for (int j=0; j<num_states-1; j++){
//       output_file << x_opt[j] << '\t';
//     }

//     output_file << x_opt[num_states-1] << std::endl;
//   }
  
//   output_file.close();
//   output_file.open("/home/patrick/Research/drake/examples/robobee/input_trajopt.txt");
//   if (!output_file.is_open()) {
//       std::cerr << "Problem opening solution output file.\n";
//   }
  
//   Eigen::VectorXd u_opt;
//   for (int i = 0; i <= N; i++) {
//     tt = i/N * T;
//     u_opt = pp_utraj.value(tt);
//     output_file << tt << '\t';    
    
//     for (int j=0; j<num_input-1; j++){
//       output_file << u_opt[j] << '\t';
//     }

//     output_file << u_opt[num_input-1] << std::endl;
//   }
  
//   output_file.close();


//   output_file.close();
  
//   output_file.open("/home/patrick/Research/drake/examples/robobee/time_col_trajopt.txt");
//   if (!output_file.is_open()) {
//       std::cerr << "Problem opening solution output file.\n";
//   }
  
//   for (int j=0; j<kNumTimeSamples; j++){
//       output_file << times_col[j] << '\t';
//   }

//   output_file.close();

// //[4-2] Get the optimal state and input for all the knot points

//   Eigen::VectorXd state_col(13);                       
//   Eigen::VectorXd input_col(4);                         


//    output_file.open("/home/patrick/Research/drake/examples/robobee/state_col_trajopt.txt");
//   if (!output_file.is_open()) {
//       std::cerr << "Problem opening solution output file.\n";
//   }
//   for (int i=0; i<kNumTimeSamples; i++){
//     state_col = dircol.GetSolution(dircol.state(i));
//     for (int j=0; j<num_states-1; j++){
//         output_file << state_col[j] << '\t';
//     }
//     output_file << state_col[num_states-1] << std::endl;
//   }
//   output_file.close();

//   output_file.open("/home/patrick/Research/drake/examples/robobee/input_col_trajopt.txt");
//   if (!output_file.is_open()) {
//       std::cerr << "Problem opening solution output file.\n";
//   }
//   for (int i=0; i<kNumTimeSamples; i++){
//     input_col = dircol.GetSolution(dircol.input(i));
//     for (int j=0; j<num_input-1; j++){
//         output_file << input_col[j] << '\t';
//     }
//     output_file << input_col[num_input-1] << std::endl;
//   }
//   output_file.close();

return{0};

}  // TrajectoOptPlot

}  // namespace robobee
}  // namespace examples
}  // namespace drake
