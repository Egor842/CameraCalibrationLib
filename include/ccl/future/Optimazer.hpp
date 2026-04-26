#pragma once
#include "FitsFabric.hpp"
#include "base_optimization.hpp"
#include <array>
#include <cmath>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <vector>


namespace ccl {


/* Optimaze min function with hybrid approach of SPO (swarm particle optimization) and IDE (improved
differential evolution)
*/
class Optimazer {
private:
    std::array<double, 2> learning_coeffs = {2.0, 2.0};
    double method_change_coef = 0.5; // determines the frequency of use SPO and IDE

public:
    struct ParameterGroup {
        std::vector<size_t> indices;
        bool is_global;
    };


public:
    Optimazer(const std::array<double, 2> &learning_coeffs, const double method_change_coef)
        : learning_coeffs(learning_coeffs),
          method_change_coef(method_change_coef) {}
    Optimazer() = default;

    std::vector<double> compute_optimum(
        size_t particle_size,
        size_t count_particles,
        const FitnessFunc &fitness_func,
        std::vector<std::pair<double, double>> ranges,
        std::vector<ParameterGroup> parameter_groups,
        size_t iterations_count = 1000
    ) {
        cv::Mat population(count_particles * 2, particle_size, CV_64F);

        if (!fill_random_values(population, ranges)) {
            return std::vector<double>();
        }

        cv::Mat population_optimals = population.clone();
        size_t best_population_optimals_idx = 0;
        cv::Mat velocity = cv::Mat::zeros(cv::Size(particle_size, count_particles * 2), CV_64F);

        std::vector<double> population_fits(count_particles * 2);
        double best_population_fit = std::numeric_limits<double>::max();

        for (int idx = 0; idx < population.rows; ++idx) {
            auto fit = fitness_func(population.row(idx));
            if (!fit.has_value()) {
                return std::vector<double>();
            }

            population_fits[idx] = fit.value();
            if (fit.value() < best_population_fit) {
                best_population_fit = fit.value();
                best_population_optimals_idx = idx;
            }
        }

        std::vector<size_t> curr_indices(count_particles * 2, 0);

        for (size_t iteration = 1; iteration <= iterations_count; iteration++) {
            std::cout << iteration << std::endl;
            for (size_t idx = 0; idx < curr_indices.size(); idx++) {
                curr_indices[idx] = idx;
            }
            fisher_yates_shuffle(curr_indices);

            double judgment_factor = method_change_coef * 2.0 / M_PI;
            double argument =
                static_cast<double>(iterations_count - iteration + 1) / (static_cast<double>(iteration * iteration));
            judgment_factor *= M_PI_2 - std::atan(argument);

            cv::Mat sub_population(count_particles, particle_size, CV_64F);
            for (size_t row_idx = 0; row_idx < count_particles; row_idx++) {
                size_t curr_idx = curr_indices[row_idx];
                population.row(curr_idx).copyTo(sub_population.row(row_idx));
            }

            cv::Mat temp_population(count_particles, particle_size, CV_64F);
            for (size_t row_idx = count_particles; row_idx < count_particles * 2; row_idx++) {
                size_t curr_idx = curr_indices[row_idx];
                population.row(curr_idx).copyTo(temp_population.row(row_idx - count_particles));
            }

            double sum_weight = 0.0;
            cv::Mat weight(count_particles, particle_size, CV_64F);
            std::vector<double> w_values(count_particles);
            for (size_t row_idx = 0; row_idx < count_particles; row_idx++) {
                double w = RandomGenerator::generate_double(0.0, 1.0);
                double w_3 = w * w * w;
                weight.row(row_idx).setTo(w_3);
                sum_weight += w_3;
            }
            weight /= sum_weight;

            temp_population = temp_population.mul(weight);

            cv::Mat temp_center(1, temp_population.cols, CV_64F);
            temp_center.setTo(0);
            for (size_t i = 0; i < temp_population.rows; i++) {
                temp_center += temp_population.row(i);
            }

            std::vector<cv::Mat> mutants(parameter_groups.size());
            std::vector<int> use_spo_method(particle_size, 0);

            for (size_t group_idx = 0; group_idx < parameter_groups.size(); group_idx++) {
                const auto &group = parameter_groups[group_idx];

                mutants[group_idx] = sub_population.clone();

                double s = RandomGenerator::generate_double(0.0, 1.0);

                if (s < judgment_factor) {
                    for (size_t row_idx = 0; row_idx < count_particles; row_idx++) {
                        int random_idx = RandomGenerator::generate_int(0, count_particles - 1);
                        while (random_idx == static_cast<int>(row_idx)) {
                            random_idx = RandomGenerator::generate_int(0, count_particles - 1);
                        }

                        for (size_t param_idx : group.indices) {
                            double scale = RandomGenerator::generate_norm_double(0.0, 1.0);
                            scale = scale * scale * scale;

                            double diff =
                                temp_center.at<double>(0, param_idx) - sub_population.at<double>(random_idx, param_idx);

                            double new_val = sub_population.at<double>(row_idx, param_idx) + scale * diff;

                            if (new_val < ranges[param_idx].first) {
                                new_val = ranges[param_idx].first;
                            }
                            if (new_val > ranges[param_idx].second) {
                                new_val = ranges[param_idx].second;
                            }

                            mutants[group_idx].at<double>(row_idx, param_idx) = new_val;
                        }
                    }
                } else {
                    for (size_t row_idx = 0; row_idx < count_particles; row_idx++) {
                        size_t curr_idx = curr_indices[row_idx];

                        double r1 = RandomGenerator::generate_double(0.0, 1.0);
                        double r2 = RandomGenerator::generate_double(0.0, 1.0);

                        for (size_t param_idx : group.indices) {
                            double v = velocity.at<double>(curr_idx, param_idx) * 0.7;

                            v += learning_coeffs[0] * r1 *
                                 (population_optimals.at<double>(curr_idx, param_idx) -
                                  population.at<double>(curr_idx, param_idx));

                            v += learning_coeffs[1] * r2 *
                                 (population_optimals.at<double>(best_population_optimals_idx, param_idx) -
                                  population.at<double>(curr_idx, param_idx));

                            double max_v = (ranges[param_idx].second - ranges[param_idx].first) * 0.2;
                            if (v > max_v) {
                                v = max_v;
                            }
                            if (v < -max_v) {
                                v = -max_v;
                            }

                            velocity.at<double>(curr_idx, param_idx) = v;

                            double new_val = population.at<double>(curr_idx, param_idx) + v;

                            if (new_val < ranges[param_idx].first) {
                                new_val = ranges[param_idx].first;
                            }
                            if (new_val > ranges[param_idx].second) {
                                new_val = ranges[param_idx].second;
                            }

                            mutants[group_idx].at<double>(row_idx, param_idx) = new_val;
                        }
                    }
                }
            }

            std::vector<double> best_mutant_fits(count_particles, std::numeric_limits<double>::max());
            std::vector<int> best_mutant_indices(count_particles, -1);

            for (size_t group_idx = 0; group_idx < mutants.size(); group_idx++) {
                for (size_t row_idx = 0; row_idx < count_particles; row_idx++) {
                    auto fit = fitness_func(mutants[group_idx].row(row_idx));
                    if (!fit.has_value()) {
                        continue;
                    }

                    if (fit.value() < best_mutant_fits[row_idx]) {
                        best_mutant_fits[row_idx] = fit.value();
                        best_mutant_indices[row_idx] = group_idx;
                    }
                }
            }

            for (size_t row_idx = 0; row_idx < count_particles; row_idx++) {
                size_t curr_idx = curr_indices[row_idx];

                if (best_mutant_indices[row_idx] >= 0) {
                    int best_group = best_mutant_indices[row_idx];

                    if (best_mutant_fits[row_idx] < population_fits[curr_idx]) {
                        mutants[best_group].row(row_idx).copyTo(population.row(curr_idx));
                        population_fits[curr_idx] = best_mutant_fits[row_idx];
                        mutants[best_group].row(row_idx).copyTo(population_optimals.row(curr_idx));

                        if (best_mutant_fits[row_idx] < best_population_fit) {
                            best_population_fit = best_mutant_fits[row_idx];
                            best_population_optimals_idx = curr_idx;
                        }
                    } else {
                        if (judgment_factor <= method_change_coef) {
                            mutants[best_group].row(row_idx).copyTo(population.row(curr_idx));
                        }
                    }
                }
            }
        }

        std::vector<double> res;
        for (size_t idx = 0; idx < particle_size; idx++) {
            res.push_back(population_optimals.at<double>(best_population_optimals_idx, idx));
        }

        return res;
    }

    std::vector<double> compute_optimum(
        size_t particle_size,
        size_t count_particles,
        const FitnessFunc &fitness_func,
        std::vector<std::pair<double, double>> ranges,
        size_t iterations_count = 1000
    ) {
        cv::Mat population(count_particles * 2, particle_size, CV_64F);

        if (!fill_random_values(population, ranges)) {
            return std::vector<double>();
        }

        cv::Mat population_optimals = population.clone();
        size_t best_population_optimals_idx = 0;
        cv::Mat velocity = cv::Mat::zeros(cv::Size(particle_size, count_particles * 2), CV_64F);

        std::vector<double> population_fits(count_particles * 2);
        double best_population_fit = std::numeric_limits<double>::max();

        for (int idx = 0; idx < population.rows; ++idx) {
            auto fit = fitness_func(population.row(idx));
            if (!fit.has_value()) {
                return std::vector<double>();
            }

            population_fits[idx] = fit.value();
            if (fit.value() < best_population_fit) {
                best_population_fit = fit.value();
                best_population_optimals_idx = idx;
            }
        }

        std::vector<size_t> curr_indices(count_particles * 2, 0);

        for (size_t iteration = 1; iteration <= iterations_count; iteration++) {
            for (size_t idx = 0; idx < curr_indices.size(); idx++) {
                curr_indices[idx] = idx;
            }
            fisher_yates_shuffle(curr_indices);

            double judgment_factor = method_change_coef * 2.0 / M_PI;
            double argument =
                static_cast<double>(iterations_count - iteration + 1) / (static_cast<double>(iteration * iteration));
            judgment_factor *= M_PI_2 - std::atan(argument);

            cv::Mat sub_population(count_particles, particle_size, CV_64F);
            for (size_t row_idx = 0; row_idx < count_particles; row_idx++) {
                size_t curr_idx = curr_indices[row_idx];
                population.row(curr_idx).copyTo(sub_population.row(row_idx));
            }

            cv::Mat temp_population(count_particles, particle_size, CV_64F);
            for (size_t row_idx = count_particles; row_idx < count_particles * 2; row_idx++) {
                size_t curr_idx = curr_indices[row_idx];
                population.row(curr_idx).copyTo(temp_population.row(row_idx - count_particles));
            }

            double sum_weight = 0.0;
            cv::Mat weight(count_particles, particle_size, CV_64F);
            std::vector<double> w_values(count_particles);
            for (size_t row_idx = 0; row_idx < count_particles; row_idx++) {
                double w = RandomGenerator::generate_double(0.0, 1.0);
                double w_3 = w * w * w;
                weight.row(row_idx).setTo(w_3);
                sum_weight += w_3;
            }
            weight /= sum_weight;

            temp_population = temp_population.mul(weight);

            double alpha = RandomGenerator::generate_double(0.0, 1.0);
            double betha = RandomGenerator::generate_double(0.0, 1.0);
            double k = RandomGenerator::generate_double(0.0, 1.0);
            double H;
            if (alpha < betha) {
                H = k * k * k;
            } else {
                H = 1 - k * k * k;
            }

            std::vector<size_t> V(particle_size, 0);
            for (size_t idx = 0; idx < V.size(); idx++) {
                V[idx] = idx;
            }
            fisher_yates_shuffle(V);
            int J = static_cast<int>(std::ceil(H * particle_size));

            cv::Mat binary_map = cv::Mat::zeros(count_particles, particle_size, CV_64F);
            for (size_t row_idx = 0; row_idx < count_particles; row_idx++) {
                for (int col_idx = 0; col_idx < J; col_idx++) {
                    size_t curr_idx = V[col_idx];
                    binary_map.at<double>(row_idx, curr_idx) = 1;
                }
            }

            cv::Mat scale_factor(count_particles, particle_size, CV_64F);
            if (alpha < betha) {
                double lamda = RandomGenerator::generate_norm_double(0.0, 1.0);
                scale_factor.setTo(lamda * lamda * lamda);
            } else {
                cv::Mat lamda_mat(count_particles, 1, CV_64F);
                for (size_t curr_row = 0; curr_row < count_particles; curr_row++) {
                    double lamda = RandomGenerator::generate_norm_double(0.0, 1.0);
                    lamda_mat.at<double>(curr_row, 0) = lamda * lamda * lamda;
                }
                scale_factor = lamda_mat * cv::Mat::ones(1, particle_size, CV_64F);
            }

            cv::Mat temp_center(1, temp_population.cols, CV_64F);
            temp_center.setTo(0);
            for (size_t i = 0; i < temp_population.rows; i++) {
                temp_center += temp_population.row(i);
            }

            std::vector<uint8_t> spo_method_use(count_particles, false);
            for (size_t row_idx = 0; row_idx < count_particles; row_idx++) {
                size_t curr_idx = curr_indices[row_idx];
                double s = RandomGenerator::generate_double(0.0, 1.0);

                if (s < judgment_factor) {
                    int random_idx = RandomGenerator::generate_int(0, count_particles - 1);
                    while (random_idx == static_cast<int>(row_idx)) {
                        random_idx = RandomGenerator::generate_int(0, count_particles - 1);
                    }

                    cv::Mat scale_row = scale_factor.row(row_idx);
                    cv::Mat binary_row = binary_map.row(row_idx);
                    cv::Mat diff = temp_center - sub_population.row(random_idx);

                    cv::Mat mutation;
                    cv::multiply(scale_row, binary_row, mutation);
                    cv::multiply(mutation, diff, mutation);

                    sub_population.row(row_idx) += mutation;

                    spo_method_use[row_idx] = false;
                } else {
                    double r1 = RandomGenerator::generate_double(0.0, 1.0);
                    double r2 = RandomGenerator::generate_double(0.0, 1.0);

                    velocity.row(curr_idx) *= 0.7;
                    velocity.row(curr_idx) +=
                        learning_coeffs[0] * r1 * (population_optimals.row(curr_idx) - population.row(curr_idx));
                    velocity.row(curr_idx) +=
                        learning_coeffs[1] * r2 *
                        (population_optimals.row(best_population_optimals_idx) - population.row(curr_idx));

                    for (size_t col_idx = 0; col_idx < particle_size; ++col_idx) {
                        double &v = velocity.at<double>(curr_idx, col_idx);
                        double max_v = (ranges[col_idx].second - ranges[col_idx].first) * 0.2;
                        if (v > max_v) {
                            v = max_v;
                        }
                        if (v < -max_v) {
                            v = -max_v;
                        }
                    }

                    sub_population.row(row_idx) = population.row(curr_idx) + velocity.row(curr_idx);

                    spo_method_use[row_idx] = true;
                }

                for (size_t j = 0; j < particle_size; ++j) {
                    double &val = sub_population.at<double>(row_idx, j);
                    if (val < ranges[j].first) {
                        val = ranges[j].first;
                    }
                    if (val > ranges[j].second) {
                        val = ranges[j].second;
                    }
                }
            }

            for (size_t row_idx = 0; row_idx < count_particles; row_idx++) {
                size_t curr_idx = curr_indices[row_idx];
                auto fit = fitness_func(sub_population.row(row_idx));
                if (!fit.has_value()) {
                    return std::vector<double>();
                }

                if (fit.value() < population_fits[curr_idx]) {
                    sub_population.row(row_idx).copyTo(population.row(curr_idx));
                    population_fits[curr_idx] = fit.value();
                    sub_population.row(row_idx).copyTo(population_optimals.row(curr_idx));

                    if (fit.value() < best_population_fit) {
                        best_population_fit = fit.value();
                        best_population_optimals_idx = curr_idx;
                    }
                } else {
                    if (spo_method_use[row_idx] == true) {
                        sub_population.row(row_idx).copyTo(population.row(curr_idx));
                    }
                }
            }
        }

        std::vector<double> res;
        for (size_t idx = 0; idx < particle_size; idx++) {
            double val = population_optimals.at<double>(best_population_optimals_idx, idx);
            res.push_back(val);
        }

        return res;
    }
};


} // namespace ccl
