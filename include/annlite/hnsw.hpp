// HNSW: Hierarchical Navigable Small World graphs.
//
// The idea in one paragraph. Build a proximity graph over the vectors and
// search it greedily: from wherever you are, hop to the neighbour closest to
// the query, repeat. Greedy descent on a flat graph gets stuck in local minima,
// so stack the graph into layers where the top holds a handful of nodes with
// very long edges and each layer down is denser. Search enters at the top,
// zooms across the space in a few hops, and drops a layer each time it stops
// improving. Log-ish hops instead of a linear scan.
//
// The two details that actually decide whether it works:
//
//   1. The neighbour selection heuristic. Keeping a node's M nearest
//      neighbours seems obvious and builds a badly connected graph, because in
//      a cluster every node picks the same close-by cluster members and
//      nothing links the clusters. The heuristic instead keeps a candidate
//      only if it's closer to the node than to any already-kept neighbour,
//      which preserves the long edges that make the graph navigable.
//
//   2. Pruning on the back-link. Edges are bidirectional, so adding one can
//      push a neighbour over its degree cap, and re-running the heuristic on
//      that neighbour is what stops hub nodes from collecting thousands of
//      edges.
//
// Reference: Malkov & Yashunin, 2016 (arXiv:1603.09320).

#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "distance.hpp"

namespace annlite {

using id_t = uint32_t;
using label_t = uint64_t;

constexpr id_t INVALID_ID = static_cast<id_t>(-1);

struct HnswParams {
  size_t M = 16;                // edges per node on layers above 0
  size_t ef_construction = 200; // candidate breadth while building
  size_t ef_search = 50;        // candidate breadth while querying
  size_t max_elements = 0;      // 0 grows on demand
  uint64_t seed = 100;
  Metric metric = Metric::L2;
};

struct SearchResult {
  std::vector<label_t> labels;
  std::vector<float> distances;
  size_t distance_computations = 0;
};

// A visited set that never allocates. Bumping a generation counter is O(1)
// where clearing a bitset or hash set would be O(n) per query, and at ef=100
// the query touches a tiny fraction of the index.
class VisitedPool {
 public:
  explicit VisitedPool(size_t capacity) : marks_(capacity, 0), cur_(0) {}

  void resize(size_t capacity) { marks_.resize(capacity, 0); }

  void begin() {
    if (++cur_ == 0) {  // wrapped, the only time a real clear is needed
      std::fill(marks_.begin(), marks_.end(), 0);
      cur_ = 1;
    }
  }

  bool test_and_set(id_t id) {
    if (id >= marks_.size()) marks_.resize(id + 1, 0);
    if (marks_[id] == cur_) return true;
    marks_[id] = cur_;
    return false;
  }

 private:
  std::vector<uint32_t> marks_;
  uint32_t cur_;
};

class HnswIndex {
 public:
  HnswIndex(size_t dim, HnswParams params = {})
      : dim_(dim),
        params_(params),
        M_(params.M),
        maxM0_(params.M * 2),  // layer 0 gets double, it carries all the traffic
        ef_construction_(std::max<size_t>(params.ef_construction, params.M)),
        ef_search_(params.ef_search),
        level_scale_(1.0 / std::log(1.0 * params.M)),
        rng_(params.seed) {
    if (dim == 0) throw std::invalid_argument("dim must be > 0");
    if (params.M < 2) throw std::invalid_argument("M must be >= 2");
    if (params.max_elements) reserve(params.max_elements);
  }

  size_t dim() const { return dim_; }
  size_t size() const { return count_ - deleted_count_; }
  size_t capacity() const { return data_.size() / dim_; }
  size_t raw_size() const { return count_; }
  size_t deleted() const { return deleted_count_; }
  int max_level() const { return max_level_; }
  Metric metric() const { return params_.metric; }
  size_t ef_search() const { return ef_search_; }
  void set_ef_search(size_t ef) { ef_search_ = std::max<size_t>(ef, 1); }

  void reserve(size_t n) {
    data_.reserve(n * dim_);
    links_.reserve(n);
    levels_.reserve(n);
    labels_.reserve(n);
    tombstone_.reserve(n);
  }

  // Returns false if the label already existed and was replaced.
  bool add(const float* vec, label_t label) {
    std::unique_lock lock(mutex_);
    auto it = by_label_.find(label);
    if (it != by_label_.end()) {
      // simplest correct upsert: retire the old node, insert a fresh one. an
      // in place update would have to repair every back-link pointing at it.
      mark_deleted_locked(it->second);
      by_label_.erase(it);
    }
    insert_locked(vec, label);
    return true;
  }

  bool remove(label_t label) {
    std::unique_lock lock(mutex_);
    auto it = by_label_.find(label);
    if (it == by_label_.end()) return false;
    mark_deleted_locked(it->second);
    by_label_.erase(it);
    return true;
  }

  bool contains(label_t label) const {
    std::shared_lock lock(mutex_);
    return by_label_.count(label) > 0;
  }

  SearchResult search(const float* query, size_t k, size_t ef = 0) const {
    std::shared_lock lock(mutex_);
    return search_locked(query, k, ef ? ef : ef_search_);
  }

  // Exact scan. Used by the tests as ground truth and by the benchmark to
  // measure what recall is being traded away.
  SearchResult brute_force(const float* query, size_t k) const {
    std::shared_lock lock(mutex_);
    std::priority_queue<std::pair<float, id_t>> heap;
    size_t comps = 0;
    for (id_t i = 0; i < count_; ++i) {
      if (tombstone_[i]) continue;
      const float d = distance(query, vector_at(i), dim_, params_.metric);
      ++comps;
      if (heap.size() < k) {
        heap.emplace(d, i);
      } else if (d < heap.top().first) {
        heap.pop();
        heap.emplace(d, i);
      }
    }
    return drain(heap, comps);
  }

  std::vector<float> get_vector(label_t label) const {
    std::shared_lock lock(mutex_);
    auto it = by_label_.find(label);
    if (it == by_label_.end()) throw std::out_of_range("no such label");
    const float* v = vector_at(it->second);
    return std::vector<float>(v, v + dim_);
  }

  void save(const std::string& path) const;
  static std::unique_ptr<HnswIndex> load(const std::string& path);

  struct Stats {
    size_t elements, deleted, capacity, max_level, dim;
    size_t total_edges;
    double mean_degree_layer0;
    size_t memory_bytes;
    std::string metric, simd;
  };
  Stats stats() const;

 private:
  // ---- storage ----
  size_t dim_;
  HnswParams params_;
  size_t M_, maxM0_, ef_construction_, ef_search_;
  double level_scale_;

  std::vector<float> data_;              // count_ * dim_, row major
  std::vector<std::vector<std::vector<id_t>>> links_;  // [node][level][neighbours]
  std::vector<int> levels_;
  std::vector<label_t> labels_;
  std::vector<uint8_t> tombstone_;
  std::unordered_map<label_t, id_t> by_label_;

  id_t entry_ = INVALID_ID;
  int max_level_ = -1;
  size_t count_ = 0;
  size_t deleted_count_ = 0;

  mutable std::shared_mutex mutex_;
  mutable std::mt19937_64 rng_;

  // Searches take a shared lock and run concurrently, so the visited set
  // cannot live on the index. One pool per thread instead: reused across
  // queries on that thread, so it still costs no allocation per search.
  static VisitedPool& scratch() {
    static thread_local VisitedPool pool(1024);
    return pool;
  }

  const float* vector_at(id_t id) const { return data_.data() + size_t(id) * dim_; }
  float* vector_at(id_t id) { return data_.data() + size_t(id) * dim_; }

  float dist(id_t a, const float* q) const {
    return distance(q, vector_at(a), dim_, params_.metric);
  }

  int random_level() {
    // Levels follow an exponential decay: a node reaches level l with
    // probability (1/M)^l, so the top layers stay tiny.
    std::uniform_real_distribution<double> u(0.0, 1.0);
    double r = u(rng_);
    if (r <= 0.0) r = 1e-12;
    return static_cast<int>(-std::log(r) * level_scale_);
  }

  void mark_deleted_locked(id_t id) {
    if (!tombstone_[id]) {
      tombstone_[id] = 1;
      ++deleted_count_;
    }
  }

  void insert_locked(const float* vec, label_t label);
  SearchResult search_locked(const float* query, size_t k, size_t ef) const;

  // Greedy beam search on one layer. Returns the ef closest candidates as a
  // max-heap keyed by distance, which is what the caller wants for pruning.
  std::priority_queue<std::pair<float, id_t>> search_layer(
      const float* q, id_t entry, size_t ef, int level,
      size_t* comps, bool skip_deleted) const;

  void select_neighbours(std::priority_queue<std::pair<float, id_t>>& candidates,
                         size_t M, std::vector<id_t>& out) const;

  SearchResult drain(std::priority_queue<std::pair<float, id_t>>& heap, size_t comps) const {
    SearchResult r;
    r.distance_computations = comps;
    r.labels.resize(heap.size());
    r.distances.resize(heap.size());
    size_t i = heap.size();
    while (!heap.empty()) {
      --i;
      r.labels[i] = labels_[heap.top().second];
      r.distances[i] = heap.top().first;
      heap.pop();
    }
    return r;
  }

  friend struct HnswSerializer;
};

}  // namespace annlite
