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
constexpr double LAMBDA = 10.0;           // OPTIMIZE [1e-2, 1e4] LOG

// lasso regression
constexpr double LASSO_LAMBDA = 2e4;      // OPTIMIZE [1e3, 1e6] LOG

// explorer  // turning_cost もある
constexpr double UCB1_COEF = 100.0;       // OPTIMIZE [1e0, 1e4] LOG
constexpr double UCB1_EPS = 1.0;          // OPTIMIZE [1e-3, 1e1] LOG
constexpr double TURNING_COST_50 = 1e7;   // OPTIMIZE [1e0, 1e7] LOG
constexpr double TURNING_COST_100 = 1e1;  // OPTIMIZE [1e0, 1e5] LOG
constexpr double TURNING_COST_150 = 1e1;  // OPTIMIZE [1e0, 1e5] LOG
constexpr double TURNING_COST_200 = 1e1;  // OPTIMIZE [1e0, 1e5] LOG

// 未使用
constexpr double TURNING_COST_START = 200.0;
constexpr double TURNING_COST_A = -40.0;
constexpr double TURNING_COST_B = 1.0;


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

constexpr auto Dyx = array<Vec2<int>, 4>{Vec2<int>{ 1, 0 }, { 0, 1 }, { -1, 0 }, { 0, -1 }};


struct Input {
	Stack<Vec2<int>, 1000> S, T;  // 始点・終点
	void ReadST() {
		int sy, sx, ty, tx;
		cin >> sy >> sx >> ty >> tx;
		S.emplace(sy, sx);  T.emplace(ty, tx);
	}
};


template<typename T>
struct Graph {
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

	// O(dimension^2)
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
			for (auto x = 0; x < dimension; x++) {
				invA[y][x] -= invAu[y] * invAu[x] * inv_denom;  // invA が対称行列なので無駄がある
			}
		}
	}

	// O(dimension) かかるので注意、使う側が適宜メモ化する
	inline double GetWeight(const int& index) const {
		auto res = 0.0;
		for (auto x = 0; x < dimension; x++) {
			res += invA[index][x] * b[x];
		}
		return res;
	}
};


// ラッソ回帰
// sklearn と比べると、 lambda = データ数 * alpha にして、 fit_intercept=False にすると結果が一致する
template<int dimension, int max_n_data>
struct LassoRegression {
	Stack<array<double, dimension>, max_n_data> X;             // 説明変数  // 0 要素を直接持つのは無駄だけどまあ
	//Stack<double, max_n_data> y;                               // 目的変数  // これ別に保持しておく必要ない
	array<Stack<int, max_n_data>, dimension> nonzero_indexes;  // 各説明変数に対し、それが影響するすべてのデータのインデックスを持つ
	array<double, dimension> squared_norm;                     // 各説明変数の l2 ノルムの 2 乗

	double lambda;

	array<double, dimension> weights;
	//Stack<double, max_n_data> y_pred;
	Stack<double, max_n_data> residuals;  // 目的変数から予測値を引いた値

	LassoRegression(const double& arg_lambda) : X(), /*y(),*/ nonzero_indexes(), squared_norm(), lambda(arg_lambda), weights(), residuals() {}

	// O(X の非 0 要素数)
	inline void Iterate() {  // TODO : 枝刈り
		for (auto j = 0; j < dimension; j++) {  // 各座標方向に対して最適化
			if (squared_norm[j] == 0.0) continue;
			auto rho = 0.0;  // 最小二乗解
			const auto& old_weight = weights[j];
			for (const auto& i : nonzero_indexes[j]) {
				rho += X[i][j] * (residuals[i] + old_weight * X[i][j]);  // x_j^T r  // old_weight * X[i][j] の項は前計算で省けるけどまあ
			}  // これ 1/N しないと合計二乗誤差、1/N すると平均二乗誤差か？？？
			const auto& new_weight = SoftThreshold(rho) / (squared_norm[j]);
			if (new_weight == old_weight) continue;
			for (const auto& i : nonzero_indexes[j]) {
				residuals[i] -= (new_weight - weights[j]) * X[i][j];
			}
			weights[j] = new_weight;
		}
	}

	inline double SoftThreshold(const double& rho) const {
		if (rho < -lambda) return rho + lambda;
		else if (rho < lambda) return 0.0;
		else return rho - lambda;
	}

	inline void AddData(const array<double, dimension>& data_x, const double& data_y) {
		X.push(data_x);
		//y.push(data_y);
		residuals.push(data_y);
		for (auto j = 0; j < dimension; j++) {
			if (data_x[j] != 0.0) {
				residuals.back() -= data_x[j] * weights[j];
				nonzero_indexes[j].push(X.size() - 1);
				squared_norm[j] += data_x[j] * data_x[j];
			}
		}
	}

	inline double GetWeight(const int& index) const {
		return weights[index];
	}
};


struct UltimateEstimator {
	// Ridge regression
	RidgeRegression<60> ridge;
	array<double, 60> weight_memo;  // 辺の重みのメモ (計算に O(dimension) かかるため)
	bitset<60> already_memorized;   // 辺の重みを既にメモしたか。ターン毎に初期化
	Stack<Stack<pair<signed char, signed char>, 60>, 999> ridge_train_data;  // (使った道、使った回数)
	Stack<double, 999> ridge_estimated_distances;                            // 各学習データに対する予測値

	// LASSO regression
	constexpr static auto lasso_dimension = 30 * 28 * 2 * 2;
	LassoRegression<lasso_dimension, 999> lasso;

	Graph<double> edge_costs;  // 各辺の予測値

	UltimateEstimator(const double& ridge_lambda, const double& lasso_lambda) :
		ridge(ridge_lambda), weight_memo(), already_memorized(), ridge_train_data(), ridge_estimated_distances(),
		lasso(lasso_lambda), edge_costs() {}

	inline int GetRidgeIndex(const bool& horizontal, const Vec2<int>& p) const {
		return horizontal ? 30 + p.y : p.x;
	}

	inline int GetLassoIndex(const bool& horizontal, const Vec2<int>& p, const bool& right) const {
		if (horizontal) {
			ASSERT_RANGE(p.y, 0, 30);
			ASSERT_RANGE(p.x, 1, 29);
			if (right) {
				return lasso_dimension / 2 + (p.y * 2 + 1) * 28 + p.x - 1;
			}
			else {
				return lasso_dimension / 2 + (p.y * 2) * 28 + p.x - 1;
			}
		}
		else {
			ASSERT_RANGE(p.x, 0, 30);
			ASSERT_RANGE(p.y, 1, 29);
			if (right) {
				return (p.x * 2 + 1) * 28 + p.y - 1;
			}
			else {
				return (p.x * 2) * 28 + p.y - 1;
			}
		}
	}

	inline double GetRidgeCost(const bool& horizonatal_edge, const Vec2<int>& p) {
		const auto bunch_index = GetRidgeIndex(horizonatal_edge, p);
		if (already_memorized[bunch_index]) {
			return weight_memo[bunch_index];
		}
		else {
			already_memorized[bunch_index] = true;
			return weight_memo[bunch_index] = ridge.GetWeight(bunch_index) + 5000.0;
		}
	}

	inline void Step() {
		ASSERT_RANGE(Info::turn, 1, 1000);
		const auto& path = Info::paths[Info::turn - 1];
		const auto& observed_distance = Info::results[Info::turn - 1];
		auto estimated_distance = 0.0;
		auto p = input.S[Info::turn - 1];
		auto data_x = array<double, 60>();
		const auto data_y = observed_distance - 5000.0 * (double)path.size();
		static auto lasso_data_x = array<double, lasso_dimension>();
		fill(lasso_data_x.begin(), lasso_data_x.end(), 0.0);
		//auto lasso_data_y = 
		ASSERT(data_x[0] == 0.0, "not initialized");
		for (const auto& d : path) {
			switch (d) {
			case Direction::D:
				data_x[GetRidgeIndex(false, p)]++;
				for (auto i = 1; i <= 28; i++) {
					lasso_data_x[GetLassoIndex(false, { i, p.x }, i <= p.y)]++;  // めちゃくちゃバグがあっても気が付かなさそうでこわい
				}
				p.y++;
				break;
			case Direction::R:
				data_x[GetRidgeIndex(true, p)]++;
				for (auto i = 1; i <= 28; i++) {
					lasso_data_x[GetLassoIndex(true, { p.y, i }, i <= p.x)]++;
				}
				p.x++;
				break;
			case Direction::U:
				p.y--;
				data_x[GetRidgeIndex(false, p)]++;
				for (auto i = 1; i <= 28; i++) {
					lasso_data_x[GetLassoIndex(false, { i, p.x }, i <= p.y)]++;
				}
				break;
			case Direction::L:
				p.x--;
				data_x[GetRidgeIndex(true, p)]++;
				for (auto i = 1; i <= 28; i++) {
					lasso_data_x[GetLassoIndex(true, { p.y, i }, i <= p.x)]++;
				}
				break;
			}
		}
		ASSERT(data_x.size() < numeric_limits<signed char>::max(), "data size error");
		ridge_train_data.emplace();
		for (auto col = (signed char)0; col < data_x.size(); col++) {
			if (data_x[col] != 0.0) ridge_train_data.back().emplace(col, (signed char)data_x[col]);
		}
		ridge.AddData(data_x, data_y);
		already_memorized.reset();
		for (auto road_index = 0; road_index < 60; road_index++) {
			weight_memo[road_index] = 5000.0 + ridge.GetWeight(road_index);
		}
		already_memorized.set();

		// lasso の目的変数の変更とデータ追加
		for (auto turn = 0; turn < Info::turn; turn++) {
			auto ridge_estimated_distance = 0.0;
			for (const auto& road_times : ridge_train_data[turn]) {
				const auto& road = road_times.first;
				const auto& times = road_times.second;
				ridge_estimated_distance += weight_memo[road] * (double)times;
			}
			if (turn != Info::turn - 1) {
				lasso.residuals[turn] -= ridge_estimated_distance - ridge_estimated_distances[turn];
				ridge_estimated_distances[turn] = ridge_estimated_distance;
			}
			else {
				lasso.AddData(lasso_data_x, observed_distance - ridge_estimated_distance);
				ridge_estimated_distances.push(ridge_estimated_distance);
			}
		}

		// lasso の重み更新
		lasso.Iterate();

		// 最終的なコスト予測
		for (auto y = 0; y < 30; y++) {  // 横
			const auto& ridge_cost = GetRidgeCost(true, { y, 0 });
			auto lasso_cost = ridge_cost;
			edge_costs.horizontal_edges[y][0] = lasso_cost;
			for (auto x = 1; x <= 28; x++) {
				lasso_cost += lasso.GetWeight(GetLassoIndex(true, { y, x }, true));
				edge_costs.horizontal_edges[y][x] = lasso_cost;  // ここもバグがこわい…
			}
			lasso_cost = 0.0;
			for (auto x = 28; x >= 1; x--) {
				lasso_cost += lasso.GetWeight(GetLassoIndex(true, { y, x }, false));
				edge_costs.horizontal_edges[y][x - 1] += lasso_cost;
			}
		}
		for (auto x = 0; x < 30; x++) {  // 縦
			const auto& ridge_cost = GetRidgeCost(false, { 0, x });
			auto lasso_cost = ridge_cost;
			edge_costs.vertical_edges[0][x] = lasso_cost;
			for (auto y = 1; y <= 28; y++) {
				lasso_cost += lasso.GetWeight(GetLassoIndex(false, { y, x }, true));
				edge_costs.vertical_edges[y][x] = lasso_cost;
			}
			lasso_cost = 0.0;
			for (auto y = 28; y >= 1; y--) {
				lasso_cost += lasso.GetWeight(GetLassoIndex(false, { y, x }, false));
				edge_costs.vertical_edges[y - 1][x] += lasso_cost;
			}
		}

	}

	inline double GetCost(const bool& horizonatal_edge, const Vec2<int>& p) {
		if (horizonatal_edge) {
			return edge_costs.horizontal_edges[p.y][p.x];
		}
		else {
			return edge_costs.vertical_edges[p.y][p.x];
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


struct Explorer {
	struct Node {
		signed char y, x;
		bool h;
	};
	UltimateEstimator* state;
	array<array<array<double, 2>, 30>, 30> distances;
	array<array<array<Node, 2>, 30>, 30> from;
	Explorer(UltimateEstimator& arg_state) : state(&arg_state), distances(), from() {}

	// 
	void Step() {
		// ダイクストラで最短路を見つける
		//const auto turning_cost = monotonic_function(TURNING_COST_START, 0.0, TURNING_COST_A, TURNING_COST_B, (double)Info::turn / 999.0);  //
		const auto turning_cost
			= Info::turn < 50 ? TURNING_COST_50
			: Info::turn < 100 ? TURNING_COST_100
			: Info::turn < 150 ? TURNING_COST_150
			: Info::turn < 200 ? TURNING_COST_200
			: 0.0;
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
			}
			else {
				if (frm.x < p.x) path.push(Direction::R);
				else path.push(Direction::L);
			}
			p = frm;
		}
		reverse(path.begin(), path.end());
	}

	inline double UCB1(const int& n) {
		// log は無視
		return UCB1_COEF / sqrt((double)n + UCB1_EPS) * (1.0 - Info::next_score_coef);
	}
};


struct Solver {
	//State state;
	//Estimator estimator;
	UltimateEstimator estimator;
	Explorer explorer;

	Solver() : estimator(LAMBDA, LASSO_LAMBDA), explorer(estimator) {}

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

struct LocalTester {
	Stack<int, 1000> A;           // 最短路長
	Stack<double, 1000> E;        // ランダマイズ係数
	Graph<int> G;                      // 正解のコスト
	LocalTester() = default;
	void ReadHV() {
		for (int y = 0; y < 30; y++) {
			for (int x = 0; x < 29; x++) {
				cin >> G.horizontal_edges[y][x];
			}
		}
		for (int y = 0; y < 29; y++) {
			for (int x = 0; x < 30; x++) {
				cin >> G.vertical_edges[y][x];
			}
		}
	}
	void ReadAE() {
		int a;
		double e;
		cin >> a >> e;
		A.push(a);  E.push(e);
	}
	int ComputePathLength(const string& path) {
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
		solver.estimator.Print();
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


namespace Test {
	void lasso_test() {
		const auto lambda = 6;
		auto lasso = LassoRegression<5, 6>(lambda);
		const auto X = array<array<double, 5>, 6>{
			array<double, 5>
			{ 0, 1, 2, 3, 4 },
			{ 5, 6, 7, 8, 9 },
			{ 3, 2, 7, 0, 2 },
			{ 9, 2, 5, 3, 1 },
			{ 0, 0, 0, 2, 5 },
			{ 9, 8, 7, 6, 1 }
		};
		const auto y = array<double, 6>{ 5, 8, 3, 2, 1, 3};
		for (auto i = 0; i < X.size(); i++) {
			lasso.AddData(X[i], y[i]);
		}
		for (auto i = 0; i < 1000; i++)lasso.Iterate();

		for (auto i = 0; i < 5; i++) {
			cout << lasso.GetWeight(i) << endl;
		}
	}
}

int main() {
	Solve();
	//Experiment::Experiment();
	//Test::lasso_test();
#ifdef _MSC_VER
	int a;
	while (1) cin >> a;
#endif
}
