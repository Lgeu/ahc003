#include<iostream>
#include<iomanip>
#include<vector>
#include<set>
#include<map>
#include<unordered_set>
#include<unordered_map>
#include<algorithm>
#include<numeric>
#include<limits>
#include<bitset>
#include<functional>
#include<type_traits>
#include<queue>
#include<stack>
#include<array>
#include<random>
#include<utility>
#include<cstdlib>
#include<ctime>
#include<string>
#include<sstream>
#include<chrono>
#include<climits>
#ifdef _MSC_VER
#include<intrin0.h>
#endif

#ifdef __GNUC__
#pragma GCC target("avx2")
#pragma GCC target("sse4")
//#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,tune=native")
#pragma GCC optimize("O3")
//#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#endif


// ========================== parameters ==========================

// ridge regression
constexpr int BUNCH = 29;  //
constexpr double LAMBDA = 10.0;

// explorer  // turning_cost もある
constexpr double UCB1_COEF = 100.0;



// ========================== macroes ==========================

#define rep(i,n) for(ll (i)=0; (i)<(n); (i)++)
#define rep1(i,n) for(ll (i)=1; (i)<=(n); (i)++)
#define rep3(i,s,n) for(ll (i)=(s); (i)<(n); (i)++)

//#define NDEBUG

#ifndef NDEBUG
#define ASSERT(expr, ...) \
		do { \
			if(!(expr)){ \
				printf("%s(%d): Assertion failed.\n", __FILE__, __LINE__); \
				printf(__VA_ARGS__); \
				abort(); \
			} \
		} while (false)
#else
#define ASSERT(...)
#endif

#define ASSERT_RANGE(value, left, right) \
    ASSERT((left <= value) && (value < right), \
		"`%s` (%d) is out of range [%d, %d)", #value, value, left, right)

#define CHECK(var) do{ std::cout << #var << '=' << var << endl; } while (false)

// ========================== utils ==========================

using namespace std;
using ll = long long;
constexpr double PI = 3.1415926535897932;

template<class T, class S> inline bool chmin(T& m, S q) {
	if (m > q) { m = q; return true; }
	else return false;
}

template<class T, class S> inline bool chmax(T& m, const S q) {
	if (m < q) { m = q; return true; }
	else return false;
}

// クリッピング  // clamp (C++17) と等価
template<class T> inline T clipped(const T& v, const T& low, const T& high) {
	return min(max(v, low), high);
}

// 2 次元ベクトル
template<typename T> struct Vec2 {
	/*
	y 軸正は下方向
	x 軸正は右方向
	回転は時計回りが正（y 軸正を上と考えると反時計回りになる）
	*/
	T y, x;
	constexpr inline Vec2() = default;
	constexpr Vec2(const T& arg_y, const T& arg_x) : y(arg_y), x(arg_x) {}
	inline Vec2(const Vec2&) = default;  // コピー
	inline Vec2(Vec2&&) = default;  // ムーブ
	inline Vec2& operator=(const Vec2&) = default;  // 代入
	inline Vec2& operator=(Vec2&&) = default;  // ムーブ代入
	template<typename S> constexpr inline Vec2(const Vec2<S>& v) : y((T)v.y), x((T)v.x) {}
	inline Vec2 operator+(const Vec2& rhs) const {
		return Vec2(y + rhs.y, x + rhs.x);
	}
	inline Vec2 operator+(const T& rhs) const {
		return Vec2(y + rhs, x + rhs);
	}
	inline Vec2 operator-(const Vec2& rhs) const {
		return Vec2(y - rhs.y, x - rhs.x);
	}
	template<typename S> inline Vec2 operator*(const S& rhs) const {
		return Vec2(y * rhs, x * rhs);
	}
	inline Vec2 operator*(const Vec2& rhs) const {  // x + yj とみなす
		return Vec2(x * rhs.y + y * rhs.x, x * rhs.x - y * rhs.y);
	}
	template<typename S> inline Vec2 operator/(const S& rhs) const {
		ASSERT(rhs != 0.0, "Zero division!");
		return Vec2(y / rhs, x / rhs);
	}
	inline Vec2 operator/(const Vec2& rhs) const {  // x + yj とみなす
		return (*this) * rhs.inv();
	}
	inline Vec2& operator+=(const Vec2& rhs) {
		y += rhs.y;
		x += rhs.x;
		return *this;
	}
	inline Vec2& operator-=(const Vec2& rhs) {
		y -= rhs.y;
		x -= rhs.x;
		return *this;
	}
	template<typename S> inline Vec2& operator*=(const S& rhs) const {
		y *= rhs;
		x *= rhs;
		return *this;
	}
	inline Vec2& operator*=(const Vec2& rhs) {
		*this = (*this) * rhs;
		return *this;
	}
	inline Vec2& operator/=(const Vec2& rhs) {
		*this = (*this) / rhs;
		return *this;
	}
	inline bool operator!=(const Vec2& rhs) const {
		return x != rhs.x || y != rhs.y;
	}
	inline bool operator==(const Vec2& rhs) const {
		return x == rhs.x && y == rhs.y;
	}
	inline void rotate(const double& rad) {
		*this = rotated(rad);
	}
	inline Vec2<double> rotated(const double& rad) const {
		return (*this) * rotation(rad);
	}
	static inline Vec2<double> rotation(const double& rad) {
		return Vec2(sin(rad), cos(rad));
	}
	static inline Vec2<double> rotation_deg(const double& deg) {
		return rotation(PI * deg / 180.0);
	}
	inline Vec2<double> rounded() const {
		return Vec2<double>(round(y), round(x));
	}
	inline Vec2<double> inv() const {  // x + yj とみなす
		const double norm_sq = l2_norm_square();
		ASSERT(norm_sq != 0.0, "Zero division!");
		return Vec2(-y / norm_sq, x / norm_sq);
	}
	inline double l2_norm() const {
		return sqrt(x * x + y * y);
	}
	inline double l2_norm_square() const {
		return x * x + y * y;
	}
	inline T l1_norm() const {
		return std::abs(x) + std::abs(y);
	}
	inline double abs() const {
		return l2_norm();
	}
	inline double phase() const {  // [-PI, PI) のはず
		return atan2(y, x);
	}
	inline double phase_deg() const {  // [-180, 180) のはず
		return phase() / PI * 180.0;
	}
};
template<typename T, typename S> inline Vec2<T> operator*(const S& lhs, const Vec2<T>& rhs) {
	return rhs * lhs;
}
template<typename T> ostream& operator<<(ostream& os, const Vec2<T>& vec) {
	os << vec.y << ' ' << vec.x;
	return os;
}

// 乱数
struct Random {
	using ull = unsigned long long;
	ull seed;
	inline Random(ull aSeed) : seed(aSeed) {
		ASSERT(seed != 0ull, "Seed should not be 0.");
	}
	const inline ull& next() {
		seed ^= seed << 9;
		seed ^= seed >> 7;
		return seed;
	}
	// (0.0, 1.0)
	inline double random() {
		return (double)next() / (double)ULLONG_MAX;
	}
	// [0, right)
	inline int randint(const int right) {
		return next() % (ull)right;
	}
	// [left, right)
	inline int randint(const int left, const int right) {
		return next() % (ull)(right - left) + left;
	}
};


// キュー
template<class T, int max_size> struct Queue {
	array<T, max_size> data;
	int left, right;
	inline Queue() : data(), left(0), right(0) {}
	inline Queue(initializer_list<T> init) :
		data(init.begin(), init.end()), left(0), right(init.size()) {}

	inline bool empty() const {
		return left == right;
	}
	inline void push(const T& value) {
		data[right] = value;
		right++;
	}
	inline void pop() {
		left++;
	}
	const inline T& front() const {
		return data[left];
	}
	template <class... Args> inline void emplace(const Args&... args) {
		data[right] = T(args...);
		right++;
	}
	inline void clear() {
		left = 0;
		right = 0;
	}
	inline int size() const {
		return right - left;
	}
};


// スタック
template<class T, int max_size> struct Stack {
	array<T, max_size> data;
	int right;

	inline Stack() : data(), right(0) {}
	inline Stack(const int n) : data(), right(0) { resize(n); }
	inline Stack(const int n, const T& val) : data(), right(0) { resize(n, val); }
	inline Stack(initializer_list<T> init) :
		data(init.begin(), init.end()), right(init.size()) {}
	inline Stack(const Stack& rhs) : data(), right(rhs.right) {  // コピー
		for (int i = 0; i < right; i++) {
			data[i] = rhs.data[i];
		}
	}
	Stack& operator=(const Stack& rhs) {
		right = rhs.right;
		for (int i = 0; i < right; i++) {
			data[i] = rhs.data[i];
		}
		return *this;
	}
	Stack& operator=(const vector<T>& rhs) {
		right = (int)rhs.size();
		ASSERT(right <= max_size, "too big vector");
		for (int i = 0; i < right; i++) {
			data[i] = rhs[i];
		}
		return *this;
	}
	Stack& operator=(Stack&&) = default;
	inline bool empty() const {
		return 0 == right;
	}
	inline void push(const T& value) {
		ASSERT_RANGE(right, 0, max_size);
		data[right] = value;
		right++;
	}
	inline T pop() {
		right--;
		ASSERT_RANGE(right, 0, max_size);
		return data[right];
	}
	const inline T& top() const {
		return data[right - 1];
	}
	template <class... Args> inline void emplace(const Args&... args) {
		ASSERT_RANGE(right, 0, max_size);
		data[right] = T(args...);
		right++;
	}
	inline void clear() {
		right = 0;
	}
	inline void resize(const int& sz) {
		ASSERT_RANGE(sz, 0, max_size + 1);
		for (; right < sz; right++) {
			data[right].~T();
			new(&data[right]) T();
		}
		right = sz;
	}
	inline void resize(const int& sz, const T& fill_value) {
		ASSERT_RANGE(sz, 0, max_size + 1);
		for (; right < sz; right++) {
			data[right].~T();
			new(&data[right]) T(fill_value);
		}
		right = sz;
	}
	inline int size() const {
		return right;
	}
	inline T& operator[](const int n) {
		ASSERT_RANGE(n, 0, right);
		return data[n];
	}
	inline const T& operator[](const int n) const {
		ASSERT_RANGE(n, 0, right);
		return data[n];
	}
	inline T* begin() {
		return (T*)data.data();
	}
	inline const T* begin() const {
		return (const T*)data.data();
	}
	inline T* end() {
		return (T*)data.data() + right;
	}
	inline const T* end() const {
		return (const T*)data.data() + right;
	}
	inline T* front() {
		ASSERT(right > 0, "no data.");
		return data[0];
	}
	inline T& back() {
		ASSERT(right > 0, "no data.");
		return data[right - 1];
	}

	inline vector<T> ToVector() {
		return vector<T>(begin(), end());
	}
};


// 時間 (秒)
inline double time() {
	return static_cast<double>(chrono::duration_cast<chrono::nanoseconds>(chrono::steady_clock::now().time_since_epoch()).count()) * 1e-9;
}


// 重複除去
template<typename T> inline void deduplicate(vector<T>& vec) {
	sort(vec.begin(), vec.end());
	vec.erase(unique(vec.begin(), vec.end()), vec.end());
}


template<typename T> inline int search_sorted(const vector<T>& vec, const T& a) {
	return lower_bound(vec.begin(), vec.end(), a) - vec.begin();
}

// popcount  // SSE 4.2 を使うべき
inline int popcount(const unsigned int& x) {
#ifdef _MSC_VER
	return (int)__popcnt(x);
#else
	return __builtin_popcount(x);
#endif
}
inline int popcount(const unsigned long long& x) {
#ifdef _MSC_VER
	return (int)__popcnt64(x);
#else
	return __builtin_popcountll(x);
#endif
}

// x >> n & 1 が 1 になる最小の n ( x==0 は未定義 )
inline int CountRightZero(const unsigned int& x) {
#ifdef _MSC_VER
	unsigned long r;
	_BitScanForward(&r, x);
	return (int)r;
#else
	return __builtin_ctz(x);
#endif
}
inline int CountRightZero(const unsigned long long& x) {
#ifdef _MSC_VER
	unsigned long r;
	_BitScanForward64(&r, x);
	return (int)r;
#else
	return __builtin_ctzll(x);
#endif
}

#ifdef _MSC_VER
inline unsigned int __builtin_clz(const unsigned int& x) { unsigned long r; _BitScanReverse(&r, x); return 31 - r; }
inline unsigned long long __builtin_clzll(const unsigned long long& x) { unsigned long r; _BitScanReverse64(&r, x); return 63 - r; }
#endif
#include<cassert>
#pragma warning( disable : 4146 )
/*
iwi 先生の radix heap (https://github.com/iwiwi/radix-heap)

The MIT License (MIT)
Copyright (c) 2015 Takuya Akiba
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
namespace radix_heap {
	namespace internal {
		template<bool Is64bit> class find_bucket_impl;

		template<>
		class find_bucket_impl<false> {
		public:
			static inline constexpr size_t find_bucket(uint32_t x, uint32_t last) {
				return x == last ? 0 : 32 - __builtin_clz(x ^ last);
			}
		};

		template<>
		class find_bucket_impl<true> {
		public:
			static inline constexpr size_t find_bucket(uint64_t x, uint64_t last) {
				return x == last ? 0 : 64 - __builtin_clzll(x ^ last);
			}
		};

		template<typename T>
		inline constexpr size_t find_bucket(T x, T last) {
			return find_bucket_impl<sizeof(T) == 8>::find_bucket(x, last);
		}

		template<typename KeyType, bool IsSigned> class encoder_impl_integer;

		template<typename KeyType>
		class encoder_impl_integer<KeyType, false> {
		public:
			typedef KeyType key_type;
			typedef KeyType unsigned_key_type;

			inline static constexpr unsigned_key_type encode(key_type x) {
				return x;
			}

			inline static constexpr key_type decode(unsigned_key_type x) {
				return x;
			}
		};

		template<typename KeyType>
		class encoder_impl_integer<KeyType, true> {
		public:
			typedef KeyType key_type;
			typedef typename std::make_unsigned<KeyType>::type unsigned_key_type;

			inline static constexpr unsigned_key_type encode(key_type x) {
				return static_cast<unsigned_key_type>(x) ^
					(unsigned_key_type(1) << unsigned_key_type(std::numeric_limits<unsigned_key_type>::digits - 1));
			}

			inline static constexpr key_type decode(unsigned_key_type x) {
				return static_cast<key_type>
					(x ^ (unsigned_key_type(1) << (std::numeric_limits<unsigned_key_type>::digits - 1)));
			}
		};

		template<typename KeyType, typename UnsignedKeyType>
		class encoder_impl_decimal {
		public:
			typedef KeyType key_type;
			typedef UnsignedKeyType unsigned_key_type;

			inline static constexpr unsigned_key_type encode(key_type x) {
				return raw_cast<key_type, unsigned_key_type>(x) ^
					((-(raw_cast<key_type, unsigned_key_type>(x) >> (std::numeric_limits<unsigned_key_type>::digits - 1))) |
						(unsigned_key_type(1) << (std::numeric_limits<unsigned_key_type>::digits - 1)));
			}

			inline static constexpr key_type decode(unsigned_key_type x) {
				return raw_cast<unsigned_key_type, key_type>
					(x ^ (((x >> (std::numeric_limits<unsigned_key_type>::digits - 1)) - 1) |
						(unsigned_key_type(1) << (std::numeric_limits<unsigned_key_type>::digits - 1))));
			}

		private:
			template<typename T, typename U>
			union raw_cast {
			public:
				constexpr raw_cast(T t) : t_(t) {}
				operator U() const { return u_; }

			private:
				T t_;
				U u_;
			};
		};

		template<typename KeyType>
		class encoder : public encoder_impl_integer<KeyType, std::is_signed<KeyType>::value> {};
		template<>
		class encoder<float> : public encoder_impl_decimal<float, uint32_t> {};
		template<>
		class encoder<double> : public encoder_impl_decimal<double, uint64_t> {};
	}  // namespace internal

	template<typename KeyType, typename EncoderType = internal::encoder<KeyType>>
	class radix_heap {
	public:
		typedef KeyType key_type;
		typedef EncoderType encoder_type;
		typedef typename encoder_type::unsigned_key_type unsigned_key_type;

		radix_heap() : size_(0), last_(), buckets_() {
			buckets_min_.fill(std::numeric_limits<unsigned_key_type>::max());
		}

		void push(key_type key) {
			const unsigned_key_type x = encoder_type::encode(key);
			assert(last_ <= x);
			++size_;
			const size_t k = internal::find_bucket(x, last_);
			buckets_[k].emplace_back(x);
			buckets_min_[k] = std::min(buckets_min_[k], x);
		}

		key_type top() {
			pull();
			return encoder_type::decode(last_);
		}

		void pop() {
			pull();
			buckets_[0].pop_back();
			--size_;
		}

		size_t size() const {
			return size_;
		}

		bool empty() const {
			return size_ == 0;
		}

		void clear() {
			size_ = 0;
			last_ = key_type();
			for (auto& b : buckets_) b.clear();
			buckets_min_.fill(std::numeric_limits<unsigned_key_type>::max());
		}

		void swap(radix_heap<KeyType, EncoderType>& a) {
			std::swap(size_, a.size_);
			std::swap(last_, a.last_);
			buckets_.swap(a.buckets_);
			buckets_min_.swap(a.buckets_min_);
		}

	private:
		size_t size_;
		unsigned_key_type last_;
		std::array<std::vector<unsigned_key_type>,
			std::numeric_limits<unsigned_key_type>::digits + 1> buckets_;
		std::array<unsigned_key_type,
			std::numeric_limits<unsigned_key_type>::digits + 1> buckets_min_;

		void pull() {
			assert(size_ > 0);
			if (!buckets_[0].empty()) return;

			size_t i;
			for (i = 1; buckets_[i].empty(); ++i);
			last_ = buckets_min_[i];

			for (unsigned_key_type x : buckets_[i]) {
				const size_t k = internal::find_bucket(x, last_);
				buckets_[k].emplace_back(x);
				buckets_min_[k] = std::min(buckets_min_[k], x);
			}
			buckets_[i].clear();
			buckets_min_[i] = std::numeric_limits<unsigned_key_type>::max();
		}
	};

	template<typename KeyType, typename ValueType, typename EncoderType = internal::encoder<KeyType>>
	class pair_radix_heap {
	public:
		typedef KeyType key_type;
		typedef ValueType value_type;
		typedef EncoderType encoder_type;
		typedef typename encoder_type::unsigned_key_type unsigned_key_type;

		pair_radix_heap() : size_(0), last_(), buckets_() {
			buckets_min_.fill(std::numeric_limits<unsigned_key_type>::max());
		}

		void push(key_type key, const value_type& value) {
			const unsigned_key_type x = encoder_type::encode(key);
			assert(last_ <= x);
			++size_;
			const size_t k = internal::find_bucket(x, last_);
			buckets_[k].emplace_back(x, value);
			buckets_min_[k] = std::min(buckets_min_[k], x);
		}

		void push(key_type key, value_type&& value) {
			const unsigned_key_type x = encoder_type::encode(key);
			assert(last_ <= x);
			++size_;
			const size_t k = internal::find_bucket(x, last_);
			buckets_[k].emplace_back(x, std::move(value));
			buckets_min_[k] = std::min(buckets_min_[k], x);
		}

		template <class... Args>
		void emplace(key_type key, Args&&... args) {
			const unsigned_key_type x = encoder_type::encode(key);
			assert(last_ <= x);
			++size_;
			const size_t k = internal::find_bucket(x, last_);
			buckets_[k].emplace_back(std::piecewise_construct,
				std::forward_as_tuple(x), std::forward_as_tuple(args...));
			buckets_min_[k] = std::min(buckets_min_[k], x);
		}

		key_type top_key() {
			pull();
			return encoder_type::decode(last_);
		}

		value_type& top_value() {
			pull();
			return buckets_[0].back().second;
		}

		void pop() {
			pull();
			buckets_[0].pop_back();
			--size_;
		}

		size_t size() const {
			return size_;
		}

		bool empty() const {
			return size_ == 0;
		}

		void clear() {
			size_ = 0;
			last_ = key_type();
			for (auto& b : buckets_) b.clear();
			buckets_min_.fill(std::numeric_limits<unsigned_key_type>::max());
		}

		void swap(pair_radix_heap<KeyType, ValueType, EncoderType>& a) {
			std::swap(size_, a.size_);
			std::swap(last_, a.last_);
			buckets_.swap(a.buckets_);
			buckets_min_.swap(a.buckets_min_);
		}

	private:
		size_t size_;
		unsigned_key_type last_;
		std::array<std::vector<std::pair<unsigned_key_type, value_type>>,
			std::numeric_limits<unsigned_key_type>::digits + 1> buckets_;
		std::array<unsigned_key_type,
			std::numeric_limits<unsigned_key_type>::digits + 1> buckets_min_;

		void pull() {
			assert(size_ > 0);
			if (!buckets_[0].empty()) return;

			size_t i;
			for (i = 1; buckets_[i].empty(); ++i);
			last_ = buckets_min_[i];

			for (size_t j = 0; j < buckets_[i].size(); ++j) {
				const unsigned_key_type x = buckets_[i][j].first;
				const size_t k = internal::find_bucket(x, last_);
				buckets_[k].emplace_back(std::move(buckets_[i][j]));
				buckets_min_[k] = std::min(buckets_min_[k], x);
			}
			buckets_[i].clear();
			buckets_min_[i] = std::numeric_limits<unsigned_key_type>::max();
		}
	};
}  // namespace radix_heap


// ---------------------------------------------------------
//  annealing
// ---------------------------------------------------------

// 山登り (最小化)
template<class State> struct HillClimbing {
	State* state;  // score, next_score, GetCandidate(const double&), Do(), operator=(State&) の実装が必要。operator= が必要なので、State のメンバにはポインタを置かないほうが楽

	inline HillClimbing(State& arg_state) : state(&arg_state) {}

	// 呼ばれる前の state は正常 (スコアが正しいなど) である必要がある
	void optimize(const double time_limit) {
		const double t0 = time();
		int iteration = 0;
		while (true) {
			iteration++;
			const double t = time() - t0;
			if (t > time_limit) break;
			const double progress_rate = t / time_limit;
			state->GetCandidate(progress_rate);
			if (state->next_score < state->score) {
				// 改善していたら遷移する
				state->Do();
				//cout << "improved! new_score=" << new_score << " progress=" << progress_rate << endl;
			}
		}
	}
};

// 焼きなまし (最小化)
template<class State> struct SimulatedAnnealing {
	State* state;  // score, Update(const double&), Undo(), operator=(State&) の実装が必要。operator= が必要なので、State のメンバにはポインタを置かないほうが楽
	Random* rng;
	double best_score;
	State best_state;  // TODO: これなくしたほうがいいかも？

	inline SimulatedAnnealing(State& arg_state, Random& arg_rng) :
		state(&arg_state), rng(&arg_rng), best_score() {}

	// 呼ばれる前の state は正常 (スコアが正しいなど) である必要がある
	template<double (*temperature_schedule)(const double&)> void optimize(const double time_limit) {
		const double t0 = time();
		best_score = state->score;
		best_state = *state;
		double old_score = state->score;
		int iteration = 0;
		while (true) {
			iteration++;
			const double t = time() - t0;
			if (t > time_limit) break;
			const double progress_rate = t / time_limit;

			state->Update(progress_rate);
			const double new_score = state->score;
			if (chmin(best_score, new_score)) {
				//cout << "improved! new_score=" << new_score << " progress=" << progress_rate << endl;
				best_state = *state;  // 中にポインタがある場合などは注意する
			}
			const double gain = old_score - new_score;  // 最小化: 良くなったらプラス
			const double temperature = temperature_schedule(t);
			const double acceptance_proba = exp(gain / temperature);
			if (acceptance_proba > rng->random()) {
				// 遷移する
				old_score = new_score;
			}
			else {
				// 遷移しない (戻す)
				state->Undo();
			}
		}
		*state = best_state;  // 中にポインタがある場合などは注意する
	}
};


// シグモイド関数
inline double sigmoid(const double& a, const double& x) {
	return 1.0 / (1.0 + exp(-a * x));
}

// 単調増加関数 f: [0, 1] -> [0, 1]
inline double monotonically_increasing_function(const double& a, const double& b, const double& x) {
	ASSERT(b >= 0.0, "parameter `b` should be positive.");
	// a == 0 なら f(x) = x
	// a が大きいとひねくれる
	// b は最初に速く増えるか最後に速く増えるか
	// a は -10 ～ 10 くらいまで、 b は 0 ～ 10 くらいまで探せば良さそう

	if (a == 0) return x;
	const double x_left = a > 0 ? -b - 0.5 : b - 0.5;
	const double x_right = x_left + 1.0;
	const double left = sigmoid(a, x_left);
	const double right = sigmoid(a, x_right);
	const double y = sigmoid(a, x + x_left);
	return (y - left) / (right - left);  // left とかが大きい値になると誤差がヤバイ　最悪 0 除算になる  // b が正なら大丈夫っぽい
}

// 単調な関数 f: [0, 1] -> [start, end]
inline double monotonic_function(const double& start, const double& end, const double& a, const double& b, const double& x) {
	return monotonically_increasing_function(a, b, x) * (end - start) + start;
}

// ---------------------------------------------------------
//  end library
// ---------------------------------------------------------


#ifdef ONLINE_JUDGE
#define LOCAL_TEST 0
#else
#define LOCAL_TEST 1
#endif

enum struct Direction : signed char {
	D, R, U, L
};

constexpr auto Dyx = array<Vec2<int>, 4>{Vec2<int>{1, 0}, { 0, 1 }, { -1, 0 }, { 0, -1 }};


struct Input {
	Stack<Vec2<int>, 1000> S, T;  // 始点・終点
	void ReadST() {
		int sy, sx, ty, tx;
		cin >> sy >> sx >> ty >> tx;
		S.emplace(sy, sx);  T.emplace(ty, tx);
	}
};


template<typename T>
struct Graph{
    array<array<T, 29>, 30> horizontal_edges;  // 横移動のコスト
    array<array<T, 30>, 29> vertical_edges;  // 縦移動のコスト
    
	Graph() = default;
	Graph(const T& fill_value) {
		fill(&horizontal_edges[0][0], &horizontal_edges[0][0] + sizeof(horizontal_edges) / sizeof(fill_value), fill_value);
		fill(&vertical_edges[0][0], &vertical_edges[0][0] + sizeof(vertical_edges) / sizeof(fill_value), fill_value);
	}

	T ComputePathLength(Vec2<int> p, const string& path) const {
		auto res = (T)0;
		for (const auto& c : path) {
			if (c == 'D') {
				res += vertical_edges[p.y][p.x];
				p.y++;
			}
			else if (c == 'R') {
				res += horizontal_edges[p.y][p.x];
				p.x++;
			}
			else if (c == 'U') {
				p.y--;
				res += vertical_edges[p.y][p.x];
			}
			else if (c == 'L') {
				p.x--;
				res += horizontal_edges[p.y][p.x];
			}
			else {
				ASSERT(false, "invalid direction.");
			}
		}
		return res;
	}
};

auto rng = Random(42);
auto input = Input();
namespace Info {
	constexpr auto TIME_LIMIT = 2.0;
	auto t0 = time();
	auto turn = 0;                                                               // 0-999
	auto next_score_coef = 0.0003129370833884096;                                // 0.998 ^ (999-turn)
	auto results = Stack<double, 1000>();                                        // 実際の所要時間
	auto paths = Stack<Stack<Direction, 1000>, 1000>();                          // 過去に出力したパス
	auto n_tried = Graph<int>(0);                                                // その辺を何回通ったか
	auto horizontal_edge_to_turns = array<array<Stack<short, 1000>, 29>, 30>();  // 辺を入れると、その辺を通ったターンを返してくれる
	auto vertical_edge_to_turns = array<array<Stack<short, 1000>, 30>, 29>();    // 辺を入れると、その辺を通ったターンを返してくれる
	auto horizontal_road_to_turns = array<Stack<pair<short, unsigned int>, 1000>, 30>();  // 道を入れると、その辺を通ったターンと通った辺を返してくれる
	auto vertical_road_to_turns = array<Stack<pair<short, unsigned int>, 1000>, 30>();    // 道を入れると、その辺を通ったターンと通った辺を返してくれる
}


template<int dimension>
struct RidgeRegression {
	// w = (X^T X + λI)^{-1} X^T y
	// A := X^T X + λI
	// b := X^T y
	// 逆行列を陽に持つ方法、本当は良くなさそう

	array<array<double, dimension>, dimension> invA;
	array<double, dimension> b;
	double lambda;

	array<double, dimension> invAu;  // A^{-1} u

	RidgeRegression(const double& arg_lambda) : invA(), b(), lambda(arg_lambda), invAu() {
		ASSERT(invA[1][0] == 0.0, "not initialized!");
		for (auto i = 0; i < dimension; i++) invA[i][i] = 1.0 / lambda;
	}

	inline void AddData(const array<double, dimension>& data_x, const double& data_y) {
		auto denom = 1.0;
		fill(invAu.begin(), invAu.end(), 0.0);
		for (auto y = 0; y < dimension; y++) {
			if (data_x[y] == 0.0) continue;
			for (auto x = 0; x < dimension; x++) {
				invAu[x] += invA[y][x] * data_x[y];
			}
			denom += data_x[y] * invAu[y];
			b[y] += data_x[y] * data_y;
		}
		auto inv_denom = 1.0 / denom;
		for (auto y = 0; y < dimension; y++) {
			for(auto x=0; x < dimension; x++) {
				invA[y][x] -= invAu[y] * invAu[x] * inv_denom;  // invA が対称行列なので無駄がある
			}
		}
	}

	// O(dimension) かかるので注意、使う側が適宜メモ化する
	inline double GetWeight(const int& index) {
		auto res = 0.0;
		for (auto x = 0; x < dimension; x++) {
			res += invA[index][x] * b[x];
		}
		return res;
	}
};


struct State {
	// ターン毎に a を増加するように強制するのもありかも？いやどうかな…
	struct Sigmoid {
		double a, left, right, center;
		Sigmoid() : a(0.2), left(5000.0), right(5000.0), center(14.0) {}  // 辺は 29 本なので (0+28)/2=14
		double f(const double& x) const {
			return sigmoid(a, center + x) * (right - left) + left;
		}
	};

	array<Sigmoid, 30> H, V;
	Stack<double, 999> estimated_path_distances;  // 各ターンの推定距離
	double score;

	// Do に必要な情報
	int next_road;
	Sigmoid next_value;
	double next_score;

	// Undo に必要な情報
	/*
	Sigmoid* last_changed;
	Sigmoid last_value;
	double last_score;
	*/

	State() : H(), V(), score(0.0),
		next_road(), next_value(), next_score()
		//last_changed(), last_value(), last_score()
	{
		ASSERT(H[0].left == 5000.0, "not initialized!");
	}

	void Step() {
		ASSERT_RANGE(Info::turn, 1, 1000);
		const auto& path = Info::paths[Info::turn - 1];
		const auto& observed_distance = Info::results[Info::turn - 1];
		auto estimated_distance = 0.0;
		auto p = input.S[Info::turn - 1];
		for (const auto& d : path) {
			switch (d) {
			case Direction::D:
				estimated_distance += GetCost(false, p);
				p.y++;
				break;
			case Direction::R:
				estimated_distance += GetCost(true, p);
				p.x++;
				break;
			case Direction::U:
				p.y--;
				estimated_distance += GetCost(false, p);
				break;
			case Direction::L:
				p.x--;
				estimated_distance += GetCost(true, p);
				break;
			}
		}
		const auto estimated_e = observed_distance / estimated_distance;
		score += (estimated_e - 1.0) * (estimated_e - 1.0);  // これ、2 乗じゃなくて 4 乗とかにしたほうが近似としては良さそう
		estimated_path_distances.push(estimated_distance);
	}

	void GetCandidate(const double& progress_rate) {
		// TODO: 使ったこと無い道ならやめる
		//last_score = score;
		next_score = score;
		next_road = rng.randint(60);
		const auto next_ptr = next_road >= 30 ? &H[next_road - 30] : &V[next_road];
		//last_value = *last_changed;
		next_value = *next_ptr;
		const auto r = rng.randint(4);
		switch (r) {
		case 0:  // a
			next_value.a *= exp((rng.random() - 0.5) * 2.0);
			break;
		case 1:  // center
			next_value.center = clipped(next_value.center + (rng.random() - 0.5) * 10.0, 0.5, 27.5);
			break;
		case 2:  // left
			next_value.left = clipped(next_value.left + (rng.random() - 0.5) * 500.0, 1100.0, 8900.0);
			break;
		case 3:  // right
			next_value.right = clipped(next_value.right + (rng.random() - 0.5) * 500.0, 1100.0, 8900.0);
			break;
		}
		if (rng.random() < 0.05) {
			// 低確率で左右反転
			swap(next_value.left, next_value.right);
			next_value.center = 28.0 - next_value.center;
		}
		
		// スコア差分計算
		array<double, 29> differences;  // 距離の差
		for (auto i = 0; i < 29; i++) differences[i] = next_value.f((double)i) - next_ptr->f((double)i);
		const auto& turn_patterns = next_road >= 30 ? Info::horizontal_road_to_turns[next_road - 30] : Info::vertical_road_to_turns[next_road];
		for (const auto& turn_pattern : turn_patterns) {
			const auto& turn = turn_pattern.first;
			auto pattern = turn_pattern.second;
			auto estimated_path_distance = estimated_path_distances[turn];
			do {
				ASSERT(pattern != 0u, "??");
				const auto idx = CountRightZero(pattern);
				estimated_path_distance += differences[idx];
				pattern ^= 1 << idx;
			} while (pattern);

			const auto& observed_distance = Info::results[turn];
			const auto estimated_e = observed_distance / estimated_path_distances[turn];
			const auto next_estimated_e = observed_distance / estimated_path_distance;
			next_score += (next_estimated_e - 1.0) * (next_estimated_e - 1.0)
				- (estimated_e - 1.0) * (estimated_e - 1.0);
		}
		

	}
	inline void Do() {
		const auto next_ptr = next_road >= 30 ? &H[next_road - 30] : &V[next_road];

		// estimated_path_distances の更新
		array<double, 29> differences;  // 距離の差
		for (auto i = 0; i < 29; i++) differences[i] = next_value.f((double)i) - next_ptr->f((double)i);
		const auto& turn_patterns = next_road >= 30 ? Info::horizontal_road_to_turns[next_road - 30] : Info::vertical_road_to_turns[next_road];
		for (const auto& turn_pattern : turn_patterns) {
			const auto& turn = turn_pattern.first;
			auto pattern = turn_pattern.second;
			do {
				ASSERT(pattern != 0u, "??");
				const auto idx = CountRightZero(pattern);
				estimated_path_distances[turn] += differences[idx];
				pattern ^= 1 << idx;
			} while (pattern);
		}

		// H, V の更新
		*next_ptr = next_value;

		// score の更新
		score = next_score;
	}

	inline double GetCost(const bool& horizonatal_edge, const Vec2<int>& p) const {
		return horizonatal_edge ? H[p.y].f((double)p.x) : V[p.x].f((double)p.y);
	}

	State& operator=(const State& rhs) {
		H = rhs.H;
		V = rhs.V;
		score = rhs.score;
		return *this;
	}
};

/*
struct State {
	Graph<double> graph;                       // 各辺の予測値
	//double D;                                  // ばらつき [100, 2000]  // D = sqrt(sum_delta / (n-1)) とかなので、必要ない
	double sum_delta;                          // すべての辺に対しての δ の 2 乗和
	int M;                                     // 道の途中で強さが変わるか {1, 2}
	array<int, 30> xs_h, xs_v;                 // どこで変わるか [1, 28]
	array<array<double, 2>, 30> H, V;          // 道の強さ基準値
	array<double, 30> sum_deltas_h, sum_deltas_v;  // 各道路の δ^2 の和
	Stack<double, 999> estimated_path_distances;  // 各ターンの推定距離
	double score;                              // 負の対数尤度 (最小化)


	// Undo に必要な情報
	int last_changed_r, last_changed_yx1, last_changed_yx21, last_changed_yx22, last_xs_value;
	double last_difference, last_HV_value_0, last_HV_value_1, last_sum_deltas_value, last_score;


	State() : graph(5000.0), sum_delta(0.0), M(2), xs_h(), xs_v(), H(), V(), sum_deltas_h(), sum_deltas_v(), estimated_path_distances(), score(0.0),
		last_changed_r(), last_changed_yx1(), last_changed_yx21(), last_changed_yx22(), last_xs_value(), last_difference(), last_HV_value_0(), last_HV_value_1(), last_sum_deltas_value(), last_score()
	{
		// xs, H, V の初期化
		fill(xs_h.begin(), xs_h.end(), 15);
		fill(xs_v.begin(), xs_v.end(), 15);
		fill(&H[0][0], &H[0][0] + sizeof(H) / sizeof(double), 5000.0);
		fill(&V[0][0], &V[0][0] + sizeof(V) / sizeof(double), 5000.0);

		// sum_deltas_h, sum_deltas_v は 0 で初期化されるよね？
	}

	void Step() {
		// 最初以外のターンの開始時に呼ばれて、新しいパスの文のスコアを加算する
		ASSERT_RANGE(Info::turn, 1, 1000);
		const auto& path = Info::paths[Info::turn - 1];
		const auto& observed_distance = Info::results[Info::turn - 1];
		auto estimated_distance = 0.0;
		auto p = input.S[Info::turn - 1];
		for (const auto& d : path) {
			switch (d) {
			case Direction::D:
				estimated_distance += graph.vertical_edges[p.y][p.x];
				p.y++;
				break;
			case Direction::R:
				estimated_distance += graph.horizontal_edges[p.y][p.x];
				p.x++;
				break;
			case Direction::U:
				p.y--;
				estimated_distance += graph.vertical_edges[p.y][p.x];
				break;
			case Direction::L:
				p.x--;
				estimated_distance += graph.horizontal_edges[p.y][p.x];
				break;
			}
		}
		const auto estimated_e = observed_distance / estimated_distance;
		score += 150.0 * (estimated_e - 1.0) * (estimated_e - 1.0);
		estimated_path_distances.push(estimated_distance);
	}

	void Update(const double& progress_rate) {
		// graph の値を変化させる
		// xs_h, xs_v, H, V, D が自動的に求まる
		static constexpr auto n = 2 * 30 * 29;  // 辺の数

		// 辺を選ぶ
		unsigned int target_edges;
		do {
			last_changed_r = rng.randint(2);
			last_changed_yx1 = rng.randint(30);  // [0, 30)
			const auto& turn_edges = (last_changed_r ? Info::horizontal_road_to_turns : Info::vertical_road_to_turns)[last_changed_yx1];
			if (turn_edges.size() == 0) continue;
			target_edges = turn_edges[rng.randint(turn_edges.size())].second;
			ASSERT(target_edges != 0u, "ok?");
		} while (false);
		do {
			last_changed_yx21 = rng.randint(1, 29);  // 区切り位置を選ぶ [1, 29)
			last_changed_yx22 = clipped(rng.randint(-15, 45), 0, 29);  // 区切り位置を選ぶ [0, 30)
			if (last_changed_yx21 > last_changed_yx22) swap(last_changed_yx21, last_changed_yx22);
		} while ((target_edges & ((1u << last_changed_yx22) - (1u << last_changed_yx21))) == 0u);

		last_difference = (rng.random() - 0.5) * 200.0;  // 変化量 要調整

		const auto change_horizontal = last_changed_r == 1;

		// 辺の重み変更
		for (auto i = last_changed_yx21; i < last_changed_yx22; i++) {
			auto& cost = change_horizontal ? graph.horizontal_edges[last_changed_yx1][i] : graph.vertical_edges[i][last_changed_yx1];
			cost += last_difference;
		}

		// xs ... 偏差平方和 (δ^2 の和) が小さくなるように 28 通り全探索
		auto r_sum_square_cost = 0.0;
		auto r_sum_cost = 0.0;
		for (auto i = 0; i < 29; i++) {
			const auto& cost = change_horizontal ? graph.horizontal_edges[last_changed_yx1][i] : graph.vertical_edges[i][last_changed_yx1];
			r_sum_square_cost += cost * cost;
			r_sum_cost += cost;
		}
		auto l_sum_square_cost = 0.0;
		auto l_sum_cost = 0.0;
		auto mi = 1e300;  // その道路の偏差平方和 (δ^2 の和) の最小値
		auto ami = -100;
		auto best_l_sum_cost = -100.0;
		auto best_r_sum_cost = -100.0;
		for (auto i = 1; i <= 28; i++) {
			const auto& cost = change_horizontal ? graph.horizontal_edges[last_changed_yx1][i] : graph.vertical_edges[i][last_changed_yx1];
			l_sum_square_cost += cost * cost;
			l_sum_cost += cost;
			r_sum_square_cost -= cost * cost;
			r_sum_cost -= cost;
			const auto l_sum_square_deviation = l_sum_square_cost - l_sum_cost * l_sum_cost / (double)i;  // 偏差平方和
			const auto r_sum_square_deviation = r_sum_square_cost - r_sum_cost * r_sum_cost / (double)(29 - i);  // 偏差平方和
			if (chmin(mi, l_sum_square_deviation + r_sum_square_deviation)) {
				ami = i;
				best_l_sum_cost = l_sum_cost;
				best_r_sum_cost = r_sum_cost;
			}
		}
		// H, V ... その道路の平均値
		auto& xs = change_horizontal ? xs_h : xs_v;  // どこで変わるか [1, 28]
		auto& HV = change_horizontal ? H : V;
		last_xs_value = xs[last_changed_yx1];
		last_HV_value_0 = HV[last_changed_yx1][0];
		last_HV_value_1 = HV[last_changed_yx1][1];
		xs[last_changed_yx1] = ami;
		HV[last_changed_yx1][0] = best_l_sum_cost / (double)ami;
		HV[last_changed_yx1][1] = best_r_sum_cost / (double)(29 - ami);

		// D ... 不偏分散の平方根 sum_辺 δ^2 / (n-1) とする
		auto& sum_deltas = change_horizontal ? sum_deltas_h : sum_deltas_v;  // 各道路の δ^2 の和
		last_sum_deltas_value = sum_deltas[last_changed_yx1];
		sum_deltas[last_changed_yx1] = mi;
		const auto old_sum_delta = sum_delta;
		sum_delta += mi - last_sum_deltas_value;

		// スコア差分計算 (δ の寄与)
		last_score = score;
		score += (double)n * 0.5 * log(
			max(sum_delta, (double)(100*100*(n-1)))  // D >= 100 であるから、 sum δ^2 >= 100^2 (n-1)
			/ max(old_sum_delta, (double)(100*100*(n-1)))
		);

		// スコア差分計算 (e の寄与)
		auto diff_negative_log_likilihood_by_e = 0.0;
		const auto mask = (1u << last_changed_yx22) - (1u << last_changed_yx21);
		const auto& turn_patterns = change_horizontal ? Info::horizontal_road_to_turns[last_changed_yx1] : Info::vertical_road_to_turns[last_changed_yx1];
		for (const auto& turn_pattern : turn_patterns) {
			const auto& turn = turn_pattern.first;
			const auto& pattern = turn_pattern.second;
			const auto n_used_edges = popcount(mask & pattern);

			// 古いものを引く
			const auto& old_estimated_path_distance = estimated_path_distances[turn];
			const auto old_estimated_e = Info::results[turn] - old_estimated_path_distance;
			diff_negative_log_likilihood_by_e -= (old_estimated_e - 1.0) * (old_estimated_e - 1.0);

			// 新しいものを足す
			estimated_path_distances[turn] += last_difference * (double)n_used_edges;  // ほとんど戻すことになることを考えるとこれは無駄…まあいいか
			const auto& estimated_path_distance = estimated_path_distances[turn];
			const auto estimated_e = Info::results[turn] - estimated_path_distance;
			diff_negative_log_likilihood_by_e += (estimated_e - 1.0) * (estimated_e - 1.0);
		}
		diff_negative_log_likilihood_by_e *= 150.0;
		score += diff_negative_log_likilihood_by_e;

	}
	void Undo() {
		const auto change_horizontal = last_changed_r == 1;

		score = last_score;

		// estimated_path_distances の復元
		const auto mask = (1u << last_changed_yx22) - (1u << last_changed_yx21);
		const auto& turn_patterns = change_horizontal ? Info::horizontal_road_to_turns[last_changed_yx1] : Info::vertical_road_to_turns[last_changed_yx1];
		for (const auto& turn_pattern : turn_patterns) {
			const auto& turn = turn_pattern.first;
			const auto& pattern = turn_pattern.second;
			const auto n_used_edges = popcount(mask & pattern);
			estimated_path_distances[turn] -= last_difference * (double)n_used_edges;  // ほとんど戻すことになることを考えるとこれは無駄…まあいいか
		}

		// sum_delta, sum_delta の復元
		auto& sum_deltas = change_horizontal ? sum_deltas_h : sum_deltas_v;
		sum_delta += last_sum_deltas_value - sum_deltas[last_changed_yx1];
		sum_deltas[last_changed_yx1] = last_sum_deltas_value;

		// xs, HV の復元
		auto& xs = change_horizontal ? xs_h : xs_v;  // どこで変わるか [1, 28]
		auto& HV = change_horizontal ? H : V;
		xs[last_changed_yx1] = last_xs_value;
		HV[last_changed_yx1][0] = last_HV_value_0;
		HV[last_changed_yx1][1] = last_HV_value_1;

		// 辺の重みの復元
		for (auto i = last_changed_yx21; i < last_changed_yx22; i++) {
			auto& cost = change_horizontal ? graph.horizontal_edges[last_changed_yx1][i] : graph.vertical_edges[i][last_changed_yx1];
			cost -= last_difference;
		}
	}
	void CalcScoreNaive(const double& progress_rate = 1.0) {
		// 負の対数尤度を愚直計算
		// δ と γ は正規分布で近似する。このとき σ^2 = D^2 / 3
		//      e も正規分布で近似する。このとき σ^2 = 300
		// H は…どうしよう 影響小さいし無視でいいかな 分布を仮定しないことにする
		// 結局、(sum_辺 (logD + 3δ^2/D^2)) + (sum_パス 150*(e-1)^2)
		// と見せかけて、δ は D で表せて定数項になるので、結局 (2*30*29 logD) + (sum_パス 150*(e-1)^2)
		// D を直接扱うのは手間なので、 (2*30*29 * 0.5 log sum_deltas) + (sum_パス 150*(e-1)^2)

		auto negative_log_likelihood_by_delta = 0.0;  // 負の対数尤度のうち、δ と γ による寄与
		auto negative_log_likilihood_by_e = 0.0;      // 負の対数尤度のうち、e による寄与

		negative_log_likelihood_by_delta += 2 * 30 * 29 * 0.5 * log(sum_delta);

		// e
		for (auto turn = 0; turn < Info::turn; turn++) {  // <= 1000
			auto p = input.S[turn];
			const auto& path = Info::paths[turn];
			auto estimated_path_cost = 0.0;
			for (const auto& d : path) {  // 30 くらい
				switch (d) {
				case Direction::D:
					estimated_path_cost += graph.vertical_edges[p.y][p.x];
					p.y++;
					break;
				case Direction::R:
					estimated_path_cost += graph.horizontal_edges[p.y][p.x];
					p.x++;
					break;
				case Direction::U:
					p.y--;
					estimated_path_cost += graph.vertical_edges[p.y][p.x];
					break;
				case Direction::L:
					p.x--;
					estimated_path_cost += graph.horizontal_edges[p.y][p.x];
					break;
				}
			}
			const auto& actual_cost = Info::results[turn];
			const auto estimated_e = actual_cost / estimated_path_cost;
			negative_log_likilihood_by_e += (estimated_e - 1.0) * (estimated_e - 1.0);
		}
		negative_log_likilihood_by_e *= 150.0;

		score = negative_log_likelihood_by_delta + negative_log_likelihood_by_delta;
	}

	double D() const {
		return sqrt(sum_delta / (double)(2 * 30 * 29 - 1));
	}
};
*/


template<int bunch=5>
struct RidgeEstimator {
	constexpr static int bunch_per_road = (29 + bunch - 1) / bunch;
	constexpr static int ridge_dimension = bunch_per_road * 30 * 2;
	RidgeRegression<ridge_dimension> ridge;
	array<double, ridge_dimension> weight_memo;  // 辺の重みのメモ (計算に O(dimension) かかるため)
	bitset<ridge_dimension> already_memorized;   // 辺の重みを既にメモしたか。ターン毎に初期化

	RidgeEstimator(const double& lambda) : ridge(lambda), weight_memo(), already_memorized() {}

	inline int GetBunchIndex(const bool& horizontal, const Vec2<int>& p) const {
		return horizontal ? ridge_dimension / 2 + p.y * bunch_per_road + p.x / bunch
			              : p.x * bunch_per_road + p.y / bunch;
	}

	inline void Step() {
		ASSERT_RANGE(Info::turn, 1, 1000);
		const auto& path = Info::paths[Info::turn - 1];
		const auto& observed_distance = Info::results[Info::turn - 1];
		auto estimated_distance = 0.0;
		auto p = input.S[Info::turn - 1];
		auto data_x = array<double, ridge_dimension>();
		auto data_y = observed_distance - 5000.0 * (double)path.size();
		ASSERT(data_x[0] == 0.0, "not initialized");
		for (const auto& d : path) {
			switch (d) {
			case Direction::D:
				data_x[GetBunchIndex(false, p)]++;
				p.y++;
				break;
			case Direction::R:
				data_x[GetBunchIndex(true, p)]++;
				p.x++;
				break;
			case Direction::U:
				p.y--;
				data_x[GetBunchIndex(false, p)]++;
				break;
			case Direction::L:
				p.x--;
				data_x[GetBunchIndex(true, p)]++;
				break;
			}
		}
		ridge.AddData(data_x, data_y);
		already_memorized.reset();
	}

	inline double GetCost(const bool& horizonatal_edge, const Vec2<int>& p) {
		const auto bunch_index = GetBunchIndex(horizonatal_edge, p);
		if (already_memorized[bunch_index]) {
			return weight_memo[bunch_index];
		}
		else {
			already_memorized[bunch_index] = true;
			return weight_memo[bunch_index] = ridge.GetWeight(bunch_index) + 5000.0;
		}
	}

	void Print() {
		for (auto y = 0; y < 30; y++) {
			for (auto x = 0; x < 29; x++) {
				cout << (int)GetCost(true, { y, x }) << " ";
			}
			cout << endl;
		}
		for (auto y = 0; y < 29; y++) {
			for (auto x = 0; x < 30; x++) {
				cout << (int)GetCost(false, { y, x }) << " ";
			}
			cout << endl;
		}

	}
};

/*
struct Estimator {
	State* state;
	//SimulatedAnnealing<State> annealing;
	HillClimbing<State> hill_climbing;
	Estimator(State& arg_state) : state(&arg_state), 
	//annealing(arg_state, rng)
	hill_climbing(arg_state)
	{}
	void Step() {
		constexpr auto begin_turn = 50;
		if (Info::turn >= begin_turn && Info::turn % 10 == 0) {  // パラメータ
			const auto end_time = (Info::TIME_LIMIT - 0.1) * (double)(Info::turn - begin_turn) / (double)(1000 - begin_turn) + Info::t0;
			//annealing.optimize<Schedule>(end_time - time());
			hill_climbing.optimize(end_time - time());
		}
	}
	static double Schedule(const double& r) {
		return 1e-9;  // 山登り
	}
};
*/

struct Explorer {
	struct Node {
		signed char y, x;
		bool h;
	};
	RidgeEstimator<BUNCH>* state;
	array<array<array<double, 2>, 30>, 30> distances;
	array<array<array<Node, 2>, 30>, 30> from;
	Explorer(RidgeEstimator<BUNCH>& arg_state) : state(&arg_state), distances(), from() {}

	// 
	void Step() {
		// ダイクストラで最短路を見つける
		const auto turning_cost = Info::turn < 100 ? 1e7 : Info::turn < 300 ? 1e4 : 0.0;
		constexpr auto inf = numeric_limits<double>::max();
		fill(&distances[0][0][0], &distances[0][0][0] + sizeof(distances) / sizeof(decltype(inf)), inf);

		const auto& start = input.S[Info::turn];
		const auto& goal = input.T[Info::turn];
		distances[start.y][start.x][0] = 0.0;  // 縦
		distances[start.y][start.x][1] = 0.0;  // 横

		using i8 = signed char;
		auto q = radix_heap::pair_radix_heap<double, Node>();
		q.push(0.0, Node{ (i8)start.y, (i8)start.x, false });
		q.push(0.0, Node{ (i8)start.y, (i8)start.x, true });
		Node goal_node;
		while (true) {
			auto dist_v = q.top_key();
			auto v = q.top_value();
			q.pop();
			if (dist_v != distances[v.y][v.x][v.h]) continue;
			if (v.y == goal.y && v.x == goal.x) {
				goal_node = v;
				break;
			}
			// D
			if (v.y != (i8)29) {
				const auto u = Node{ v.y + (i8)1, v.x, false };
				const auto& cost = state->GetCost(false, Vec2<int>{ v.y, v.x });
				const auto& n_tried = Info::n_tried.vertical_edges[v.y][v.x];
				auto dist_u = dist_v + max(1000.0, cost - UCB1(n_tried));
				if (v.h != u.h) dist_u += turning_cost;
				if (dist_u < distances[u.y][u.x][u.h]) {
					distances[u.y][u.x][u.h] = dist_u;
					from[u.y][u.x][u.h] = v;
					q.push(dist_u, u);
				}
			}
			// R
			if (v.x != (i8)29) {
				const auto u = Node{ v.y, v.x + (i8)1, true };
				const auto& cost = state->GetCost(true, Vec2<int>{ v.y, v.x });
				const auto& n_tried = Info::n_tried.horizontal_edges[v.y][v.x];
				auto dist_u = dist_v + max(1000.0, cost - UCB1(n_tried));
				if (v.h != u.h) dist_u += turning_cost;
				if (dist_u < distances[u.y][u.x][u.h]) {
					distances[u.y][u.x][u.h] = dist_u;
					from[u.y][u.x][u.h] = v;
					q.push(dist_u, u);
				}
			}
			// U
			if (v.y != (i8)0) {
				const auto u = Node{ v.y - (i8)1, v.x, false };
				const auto& cost = state->GetCost(false, Vec2<int>{ u.y, u.x });
				const auto& n_tried = Info::n_tried.vertical_edges[u.y][u.x];
				auto dist_u = dist_v + max(1000.0, cost - UCB1(n_tried));
				if (v.h != u.h) dist_u += turning_cost;
				if (dist_u < distances[u.y][u.x][u.h]) {
					distances[u.y][u.x][u.h] = dist_u;
					from[u.y][u.x][u.h] = v;
					q.push(dist_u, u);
				}
			}
			// L
			if (v.x != (i8)0) {
				const auto u = Node{ v.y, v.x - (i8)1, true };
				const auto& cost = state->GetCost(true, Vec2<int>{ u.y, u.x });
				const auto& n_tried = Info::n_tried.horizontal_edges[u.y][u.x];
				auto dist_u = dist_v + max(1000.0, cost - UCB1(n_tried));
				if (v.h != u.h) dist_u += turning_cost;
				if (dist_u < distances[u.y][u.x][u.h]) {
					distances[u.y][u.x][u.h] = dist_u;
					from[u.y][u.x][u.h] = v;
					q.push(dist_u, u);
				}
			}
		}
		// ダイクストラの復元
		Info::paths.emplace();
		auto& path = Info::paths[Info::turn];
		auto p = goal_node;
		while (p.y != start.y || p.x != start.x) {
			const auto& frm = from[p.y][p.x][p.h];
			if (p.y != frm.y) {
				if (frm.y < p.y) path.push(Direction::D);
				else path.push(Direction::U);
			} else {
				if (frm.x < p.x) path.push(Direction::R);
				else path.push(Direction::L);
			}
			p = frm;
		}
		reverse(path.begin(), path.end());
	}

	inline double UCB1(const int& n) {
		// log は無視
		return UCB1_COEF / sqrt(n + 1) * (1.0 - Info::next_score_coef);
	}
};


struct Solver {
	//State state;
	//Estimator estimator;
	RidgeEstimator<BUNCH> estimator;
	Explorer explorer;

	Solver() : estimator(LAMBDA), explorer(estimator) {}

	inline string Solve() {
		// 結果は Info::paths に格納され、文字列化したものを返す
		if (Info::turn != 0) {
			//state.Step();
			estimator.Step();
		}
		explorer.Step();
		return stringify(Info::paths[Info::turn]);
	}

	inline string stringify(const Stack<Direction, 1000>& path) {
		auto res = ""s;
		for (const auto& d : path) res += "DRUL"[(int)d];
		return res;
	}
};

struct LocalTester{
    Stack<int, 1000> A;           // 最短路長
    Stack<double, 1000> E;        // ランダマイズ係数
	Graph<int> G;                      // 正解のコスト
	LocalTester() = default;
    void ReadHV(){
        for(int y=0;y<30;y++){
            for(int x=0; x<29; x++){
                cin >> G.horizontal_edges[y][x];
            }
        }
        for(int y=0;y<29;y++){
            for(int x=0; x<30; x++){
                cin >> G.vertical_edges[y][x];
            }
        }
    }
    void ReadAE(){
        int a;
        double e;
        cin >> a >> e;
        A.push(a);  E.push(e);
    }
    int ComputePathLength(const string& path){
        auto p = Vec2<int>(input.S[Info::turn]);
		return G.ComputePathLength(p, path);
    }
};


namespace Info {
	void UpdateInfo() {
		// horizontal_edge_to_turns とかの更新
		// 最初以外のターン開始時に呼ばれる
		// State::Step と一緒にしないのはちょっと非効率かもだけど多分大丈夫

		const auto& path = paths[turn - 1];
		auto p = input.S[turn - 1];
		auto visited_vertical_edges = unordered_map<signed char, unsigned int>();
		auto visited_horizontal_edges = unordered_map<signed char, unsigned int>();
		for (const auto& d : path) {
			switch (d) {
			case Direction::D:
				n_tried.vertical_edges[p.y][p.x]++;
				vertical_edge_to_turns[p.y][p.x].push(turn - 1);
				visited_vertical_edges[(signed char)p.x] |= 1u << p.y;
				p.y++;
				break;
			case Direction::R:
				n_tried.horizontal_edges[p.y][p.x]++;
				horizontal_edge_to_turns[p.y][p.x].push(turn - 1);
				visited_horizontal_edges[(signed char)p.y] |= 1u << p.x;
				p.x++;
				break;
			case Direction::U:
				p.y--;
				n_tried.vertical_edges[p.y][p.x]++;
				vertical_edge_to_turns[p.y][p.x].push(turn - 1);
				visited_vertical_edges[(signed char)p.x] |= 1u << p.y;
				break;
			case Direction::L:
				p.x--;
				n_tried.horizontal_edges[p.y][p.x]++;
				horizontal_edge_to_turns[p.y][p.x].push(turn - 1);
				visited_horizontal_edges[(signed char)p.y] |= 1u << p.x;
				break;
			}
		}
		for (const auto& x_ys : visited_vertical_edges) {
			const auto& x = x_ys.first;
			const auto& ys = x_ys.second;
			vertical_road_to_turns[x].emplace((short)turn - 1, ys);
		}
		for (const auto& y_xs : visited_horizontal_edges) {
			const auto& y = y_xs.first;
			const auto& xs = y_xs.second;
			horizontal_road_to_turns[y].emplace((short)turn - 1, xs);
		}
	}
}


void Solve() {
	static auto local_tester = LocalTester();
	static auto solver = Solver();
	if (LOCAL_TEST) {
		local_tester.ReadHV();
	}
	auto prev_result = 0;
	auto score = 0.0;
	for (int k = 0; k < 1000; k++) {
		input.ReadST();
		if (LOCAL_TEST) {
			local_tester.ReadAE();
		}
		if (k != 0) Info::UpdateInfo();
		auto path = solver.Solve();
		cout << path << endl;
		if (LOCAL_TEST) {
			auto b = local_tester.ComputePathLength(path);
			score = score * 0.998 + (double)local_tester.A.back() / (double)b;
			prev_result = (int)(b * local_tester.E.back() + 0.5);
		}
		else {
			cin >> prev_result;
		}
		Info::results.push(prev_result);
		Info::next_score_coef /= 0.998;
		Info::turn++;
	}
	if (LOCAL_TEST) {
		cout << (int)(2312311.0 * score + 0.5) << endl;
		//solver.estimator.Print();
	}
}


// 内部の H, V が完全にわかったとした場合、スコアはどうなるか？
namespace Experiment {
	auto D = -1;
	auto M = 2;
	auto H = array<array<int, 2>, 30>();
	auto x = array<array<int, 3>, 30>();
	auto h = array<array<int, 29>, 30>();
	auto V = array<array<int, 2>, 30>();
	auto y = array<array<int, 3>, 30>();
	auto v = array<array<int, 30>, 29>();
	array<array<int, 30>, 30> distances;
	array<array<Direction, 30>, 30> from;

	int get_cost(const bool& horizontal, const Vec2<int>& p) {
		return horizontal ? h[p.y][p.x] : v[p.y][p.x];
	}
	int get_rough_cost(const bool& horizontal, const Vec2<int>& p) {
		return horizontal ? H[p.y][p.x >= x[p.y][1]] : V[p.x][p.y >= y[p.x][1]];
	}
	template<int (*get_cost)(const bool&, const Vec2<int>&)>
	Stack<Direction, 100> Dijkstra(const Vec2<int>& start, const Vec2<int>& goal) {
		constexpr auto inf = numeric_limits<int>::max();
		fill(&distances[0][0], &distances[0][0] + sizeof(distances) / sizeof(decltype(inf)), inf);

		distances[start.y][start.x] = 0.0;

		auto q = radix_heap::pair_radix_heap<int, Vec2<int>>();
		q.push(0.0, start);

		while (true) {
			auto dist_v = q.top_key();
			auto v = q.top_value();
			q.pop();
			if (dist_v != distances[v.y][v.x]) continue;
			if (v == goal) break;
			
			// D
			if (v.y != 29) {
				const auto u = Vec2<int>{ v.y + 1, v.x };
				const auto& cost = get_cost(false, v);
				auto dist_u = dist_v + cost;
				if (dist_u < distances[u.y][u.x]) {
					distances[u.y][u.x] = dist_u;
					from[u.y][u.x] = Direction::D;
					q.push(dist_u, u);
				}
			}
			// R
			if (v.x != 29) {
				const auto u = Vec2<int>{ v.y, v.x + 1 };
				const auto& cost = get_cost(true, v);
				auto dist_u = dist_v + cost;
				if (dist_u < distances[u.y][u.x]) {
					distances[u.y][u.x] = dist_u;
					from[u.y][u.x] = Direction::R;
					q.push(dist_u, u);
				}
			}
			// U
			if (v.y != 0) {
				const auto u = Vec2<int>{ v.y - 1, v.x };
				const auto& cost = get_cost(false, u);
				auto dist_u = dist_v + cost;
				if (dist_u < distances[u.y][u.x]) {
					distances[u.y][u.x] = dist_u;
					from[u.y][u.x] = Direction::U;
					q.push(dist_u, u);
				}
			}
			// L
			if (v.x != 0) {
				const auto u = Vec2<int>{ v.y, v.x - 1 };
				const auto& cost = get_cost(true, u);
				auto dist_u = dist_v + cost;
				if (dist_u < distances[u.y][u.x]) {
					distances[u.y][u.x] = dist_u;
					from[u.y][u.x] = Direction::L;
					q.push(dist_u, u);
				}
			}
		}

		auto res = Stack<Direction, 100>();
		auto p = goal;
		while (p != start) {
			const auto& d = from[p.y][p.x];
			res.push(d);
			p -= Dyx[(int)d];
		}
		reverse(res.begin(), res.end());
		return res;
	}

	template<int (*get_cost)(const bool&, const Vec2<int>&)>
	int CalculatePathDistance(const Stack<Direction, 100>& path, const Vec2<int>& start) {
		auto res = 0;
		auto p = start;
		for (const auto& d : path) {
			switch (d) {
			case Direction::D:
				res += get_cost(false, p);
				p.y += 1;
				break;
			case Direction::R:
				res += get_cost(true, p);
				p.x += 1;
				break;
			case Direction::U:
				p.y -= 1;
				res += get_cost(false, p);
				break;
			case Direction::L:
				p.x -= 1;
				res += get_cost(true, p);
				break;
			}
		}
		return res;
	}

	void Generate() {
		D = rng.randint(100, 2001);
		for (auto&& Hi : H) for (auto&& Hij : Hi) Hij = rng.randint(1000 + D, 9001 - D);
		for (auto&& xi : x) {
			xi[0] = 0;
			xi[1] = rng.randint(1, 29);
			xi[2] = 29;
		}
		for (int i = 0; i < 30; i++) {
			for (int p = 0; p < M; p++) {
				for (int j = x[i][p]; j < x[i][p + 1]; j++) {
					h[i][j] = H[i][p] + rng.randint(-D, D + 1);
				}
			}
		}
		for (auto&& Vi : V) for (auto&& Vij : Vi) Vij = rng.randint(1000 + D, 9001 - D);
		for (auto&& yi : y) {
			yi[0] = 0;
			yi[1] = rng.randint(1, 29);
			yi[2] = 29;
		}
		for (int j = 0; j < 30; j++) {
			for (int p = 0; p < M; p++) {
				for (int i = y[j][p]; i < y[j][p + 1]; i++) {
					v[i][j] = V[j][p] + rng.randint(-D, D + 1);
				}
			}
		}
	}
	pair<Vec2<int>, Vec2<int>> GetQuery() {
		do {
			auto s = Vec2<int>(rng.randint(30), rng.randint(30));
			auto t = Vec2<int>(rng.randint(30), rng.randint(30));
			if ((s - t).l1_norm() >= 10) return make_pair(s, t);
		} while (true);
	}
	void Experiment() {
		auto sum_score = 0.0;
		for (int i = 0; i < 100; i++) {
			Generate();
			auto score = 0.0;
			for (int q = 0; q < 1000; q++) {
				auto st = GetQuery();
				auto a = CalculatePathDistance<get_cost>(Dijkstra<get_cost>(st.first, st.second), st.first);
				auto b = CalculatePathDistance<get_cost>(Dijkstra<get_rough_cost>(st.first, st.second), st.first);
				ASSERT(a <= b, "omg");
				score += (double)a / (double)b;
			}
			cout << "i=" << i << " D=" << D << " score=" << score << endl;
			sum_score += score;
		}
		cout << "sum_score=" << sum_score << endl;
	}
};


int main(){
	Solve();
	//Experiment::Experiment();
#ifdef _MSC_VER
	int a;
	while (1) cin >> a;
#endif
}
