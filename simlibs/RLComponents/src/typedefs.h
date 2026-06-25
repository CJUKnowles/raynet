#ifndef TYPEDEFS_H
#define TYPEDEFS_H

#include <tuple>
#include <array>
#include <vector>
#include <type_traits>
#include <utility>
#include <cstddef>
#include <variant>
#include <cstdint>
#include <initializer_list>
#include <string>



// Define the types for observation, action and reward.

// Provide a single runtime-flexible Observation wrapper type. It stores
// observations as a dense `std::vector<double>` but can be implicitly
// constructed from common static containers such as `std::array` and
// `std::tuple` so existing `ComputeObservation` implementations that
// return tuples/arrays will continue to compile without changing call
// sites. This removes the need to recompile to switch observation
// layouts and allows multiple schemes to be used in one build.

namespace rl {

class Observation {
public:
    Observation() = default;

    using Field = std::variant<int64_t, double, bool>;

    explicit Observation(std::vector<Field> v) : data_(std::move(v)) {}
    explicit Observation(std::vector<std::pair<std::string, Field>> v) { setNamedData(v); }
    Observation(std::initializer_list<std::pair<std::string, Field>> v) { setNamedData(v); }

    // Construct from single numeric value
    template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
    Observation(T val) { pushValue(val); }

    // Construct from std::vector-like (elements of arbitrary arithmetic types)
    template <typename V, typename = typename std::enable_if<!std::is_arithmetic<V>::value>::type>
    Observation(const V &v) { data_.reserve(v.size()); for (auto &e : v) pushValue(e); }

    // Construct from std::array
    template <typename T, std::size_t N>
    Observation(const std::array<T, N> &a) { data_.reserve(N); for (const auto &e : a) pushValue(e); }

    // Construct from std::tuple
    template <typename... Ts>
    Observation(const std::tuple<Ts...> &t) { tupleToVector(t, std::index_sequence_for<Ts...>{}); }

    std::size_t size() const noexcept { return data_.size(); }
    const std::vector<Field>& data() const noexcept { return data_; }
    std::vector<Field>& data() noexcept { return data_; }
    const std::vector<std::string>& names() const noexcept { return names_; }
    bool hasNames() const noexcept { return names_.size() == data_.size(); }

    const Field& at(std::size_t i) const { return data_.at(i); }
    Field& at(std::size_t i) { return data_.at(i); }
    const std::string& nameAt(std::size_t i) const { return names_.at(i); }

    // Convenience: get numeric value as double (integrals promoted)
    double asDouble(std::size_t i) const {
        return std::visit([](auto &&v) -> double {
            using U = std::decay_t<decltype(v)>;
            if constexpr (std::is_integral<U>::value) return static_cast<double>(v);
            else if constexpr (std::is_floating_point<U>::value) return static_cast<double>(v);
            else if constexpr (std::is_same<U, bool>::value) return v ? 1.0 : 0.0;
            else return 0.0;
        }, data_.at(i));
    }

    auto begin() noexcept { return data_.begin(); }
    auto end() noexcept { return data_.end(); }
    auto begin() const noexcept { return data_.begin(); }
    auto end() const noexcept { return data_.end(); }

private:
    template <typename Fields>
    void setNamedData(const Fields &fields) {
        data_.reserve(fields.size());
        names_.reserve(fields.size());
        for (const auto &field : fields) {
            names_.push_back(field.first);
            data_.push_back(field.second);
        }
    }

    template <typename T>
    void pushValue(T v) {
        if constexpr (std::is_integral<T>::value && !std::is_same<T, bool>::value) {
            data_.emplace_back(static_cast<int64_t>(v));
        } else if constexpr (std::is_floating_point<T>::value) {
            data_.emplace_back(static_cast<double>(v));
        } else if constexpr (std::is_same<T, bool>::value) {
            data_.emplace_back(static_cast<bool>(v));
        } else {
            static_assert(std::is_arithmetic<T>::value || std::is_same<T, bool>::value, "Unsupported Observation field type");
        }
    }

    template <typename Tuple, std::size_t... I>
    void tupleToVector(const Tuple &t, std::index_sequence<I...>) {
        data_.reserve(sizeof...(I));
        (void)std::initializer_list<int>{(pushValue(std::get<I>(t)), 0)...};
    }

    std::vector<Field> data_;
    std::vector<std::string> names_;
};

} // namespace rl

using ObsType = rl::Observation;
typedef float RewardType;
typedef float ActionType;

#endif
