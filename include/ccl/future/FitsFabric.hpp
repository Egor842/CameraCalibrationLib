#pragma once
#include <cmath>
#include <concepts>
#include <exception>
#include <functional>
#include <opencv2/opencv.hpp>
#include <optional>
#include <type_traits>
#include <vector>


namespace ccl {


template <typename T>
concept Arithmetic = std::is_arithmetic_v<T> && requires(T a, T b) {
    { a - b } -> std::convertible_to<T>;
    { a * b } -> std::convertible_to<T>;
    { std::abs(a - b) } -> std::convertible_to<double>;
};


template <typename T>
concept Point2DLike = requires(T a, T b) {
    { a.x } -> std::convertible_to<double>;
    { a.y } -> std::convertible_to<double>;
    { a - b } -> std::same_as<T>;
    { cv::norm(a - b) } -> std::convertible_to<double>;
};


template <typename T>
concept VectorLike = requires(T v) {
    typename T::value_type;
    { v.size() } -> std::convertible_to<size_t>;
    { v.begin() } -> std::input_iterator;
    { v.end() } -> std::input_iterator;
};


using FitnessFunc = std::function<std::optional<double>(const cv::Mat &params)>;


struct MSE {
    template <Arithmetic T>
    static std::optional<double> compute(const std::vector<T> &pred, const std::vector<T> &target) {
        if (pred.size() != target.size()) {
            return std::nullopt;
        }

        double error = 0.0;
        for (size_t i = 0; i < target.size(); ++i) {
            auto diff = pred[i] - target[i];
            error += static_cast<double>(diff * diff);
        }
        return error / target.size();
    }

    template <Point2DLike T>
    static std::optional<double> compute(const std::vector<T> &pred, const std::vector<T> &target) {
        if (pred.size() != target.size()) {
            return std::nullopt;
        }

        double error = 0.0;
        for (size_t i = 0; i < target.size(); ++i) {
            double diff = cv::norm(pred[i] - target[i]);
            error += diff * diff;
        }
        return error / target.size();
    }

    template <VectorLike T>
    static std::optional<double> compute(const std::vector<T> &pred, const std::vector<T> &target) {
        if (pred.size() != target.size()) {
            return std::nullopt;
        }

        double total_error = 0.0;
        size_t total_count = 0;

        for (size_t i = 0; i < target.size(); ++i) {
            auto inner_error = compute(pred[i], target[i]);
            if (!inner_error.has_value()) {
                return std::nullopt;
            }

            total_error += inner_error.value() * pred[i].size();
            total_count += pred[i].size();
        }

        return total_error / total_count;
    }
};

struct MAE {
    template <Arithmetic T>
    static std::optional<double> compute(const std::vector<T> &pred, const std::vector<T> &target) {
        if (pred.size() != target.size()) {
            return std::nullopt;
        }

        double error = 0.0;
        for (size_t i = 0; i < target.size(); ++i) {
            error += static_cast<double>(std::abs(pred[i] - target[i]));
        }
        return error / target.size();
    }

    template <Point2DLike T>
    static std::optional<double> compute(const std::vector<T> &pred, const std::vector<T> &target) {
        if (pred.size() != target.size()) {
            return std::nullopt;
        }

        double error = 0.0;
        for (size_t i = 0; i < target.size(); ++i) {
            error += cv::norm(pred[i] - target[i]);
        }
        return error / target.size();
    }

    template <VectorLike T>
    static std::optional<double> compute(const std::vector<T> &pred, const std::vector<T> &target) {
        if (pred.size() != target.size()) {
            return std::nullopt;
        }

        double total_error = 0.0;
        size_t total_count = 0;

        for (size_t i = 0; i < target.size(); ++i) {
            auto inner_error = compute(pred[i], target[i]);
            if (!inner_error.has_value()) {
                return std::nullopt;
            }

            total_error += inner_error.value() * pred[i].size();
            total_count += pred[i].size();
        }

        return total_error / total_count;
    }
};

struct RMSE {
    template <Arithmetic T>
    static std::optional<double> compute(const std::vector<T> &pred, const std::vector<T> &target) {
        if (pred.size() != target.size()) {
            return std::nullopt;
        }

        double error = 0.0;
        for (size_t i = 0; i < target.size(); ++i) {
            auto diff = pred[i] - target[i];
            error += static_cast<double>(diff * diff);
        }
        return std::sqrt(error) / target.size();
    }

    template <Point2DLike T>
    static std::optional<double> compute(const std::vector<T> &pred, const std::vector<T> &target) {
        if (pred.size() != target.size()) {
            return std::nullopt;
        }

        double error = 0.0;
        for (size_t i = 0; i < target.size(); ++i) {
            double diff = cv::norm(pred[i] - target[i]);
            error += diff * diff;
        }
        return std::sqrt(error) / target.size();
    }

    template <VectorLike T>
    static std::optional<double> compute(const std::vector<T> &pred, const std::vector<T> &target) {
        if (pred.size() != target.size()) {
            return std::nullopt;
        }

        double total_error = 0.0;
        size_t total_count = 0;

        for (size_t i = 0; i < target.size(); ++i) {
            auto inner_error = compute(pred[i], target[i]);
            if (!inner_error.has_value()) {
                return std::nullopt;
            }

            total_error += inner_error.value() * pred[i].size() * pred[i].size();
            total_count += pred[i].size();
        }

        return std::sqrt(total_error) / total_count;
    }
};


struct HUBER {
    template <Arithmetic T> static double compute(const T &pred, const T &target, double delta = 1.0) {
        double diff = static_cast<double>(std::abs(pred - target));
        if (diff <= delta) {
            return 0.5 * diff * diff;
        } else {
            return delta * (diff - 0.5 * delta);
        }
    }

    template <Arithmetic T>
    static std::optional<double> compute(const std::vector<T> &pred, const std::vector<T> &target, double delta = 1.0) {
        if (pred.size() != target.size()) {
            return std::nullopt;
        }

        double error = 0.0;
        for (size_t i = 0; i < target.size(); ++i) {
            error += compute(pred[i], target[i], delta);
        }
        return error / target.size();
    }

    template <Point2DLike T> static double compute(const T &pred, const T &target, double delta = 1.0) {
        double diff = cv::norm(pred - target);
        if (diff <= delta) {
            return 0.5 * diff * diff;
        } else {
            return delta * (diff - 0.5 * delta);
        }
    }

    template <Point2DLike T>
    static std::optional<double> compute(const std::vector<T> &pred, const std::vector<T> &target, double delta = 1.0) {
        if (pred.size() != target.size()) {
            return std::nullopt;
        }

        double error = 0.0;
        for (size_t i = 0; i < target.size(); ++i) {
            error += compute(pred[i], target[i], delta);
        }
        return error / target.size();
    }

    template <VectorLike T>
    static std::optional<double> compute(const std::vector<T> &pred, const std::vector<T> &target, double delta = 1.0) {
        if (pred.size() != target.size()) {
            return std::nullopt;
        }

        double total_error = 0.0;
        size_t total_count = 0;

        for (size_t i = 0; i < target.size(); ++i) {
            auto inner_error = compute(pred[i], target[i], delta);
            if (!inner_error.has_value()) {
                return std::nullopt;
            }

            total_error += inner_error.value() * pred[i].size();
            total_count += pred[i].size();
        }

        return total_error / total_count;
    }
};


enum class Metric
{
    MSE,
    MAE,
    RMSE,
    HUBER
};


class FitnessFactoryException : public std::exception {
private:
    std::string message;

public:
    explicit FitnessFactoryException(const char *msg) : message(msg) {}
    explicit FitnessFactoryException(const std::string &msg) : message(msg) {}

    const char *what() const noexcept override {
        return message.c_str();
    }
};


class FitnessFactory {
public:
    template <typename InputType, typename OutputType>
    static FitnessFunc create(
        const std::vector<InputType> &inputs,
        const std::vector<OutputType> &outputs,
        std::function<OutputType(const cv::Mat &params, const InputType &input)> model,
        Metric metric
    ) {
        if (inputs.size() != outputs.size()) {
            throw FitnessFactoryException("inputs and outputs must have same size");
        }

        switch (metric) {
        case Metric::MSE:
            return [inputs, outputs, model](const cv::Mat &params) {
                std::vector<OutputType> predictions;
                predictions.reserve(inputs.size());
                for (const auto &input : inputs) {
                    predictions.push_back(model(params, input));
                }
                return MSE::compute(predictions, outputs);
            };

        case Metric::MAE:
            return [inputs, outputs, model](const cv::Mat &params) {
                std::vector<OutputType> predictions;
                predictions.reserve(inputs.size());
                for (const auto &input : inputs) {
                    predictions.push_back(model(params, input));
                }
                return MAE::compute(predictions, outputs);
            };

        case Metric::RMSE:
            return [inputs, outputs, model](const cv::Mat &params) {
                std::vector<OutputType> predictions;
                predictions.reserve(inputs.size());
                for (const auto &input : inputs) {
                    predictions.push_back(model(params, input));
                }
                return RMSE::compute(predictions, outputs);
            };

        case Metric::HUBER:
            return [inputs, outputs, model](const cv::Mat &params) {
                const double delta = 1.0;
                std::vector<OutputType> predictions;
                predictions.reserve(inputs.size());
                for (const auto &input : inputs) {
                    predictions.push_back(model(params, input));
                }
                return HUBER::compute(predictions, outputs, delta);
            };

        default:
            throw FitnessFactoryException("Unknown metric");
        }
    }
};


}; // namespace ccl
