#include "annlite/hnsw.hpp"

#include <cstdio>
#include <fstream>

namespace annlite {

// ---------------------------------------------------------------- search_layer

std::priority_queue<std::pair<float, id_t>> HnswIndex::search_layer(
    const float* q, id_t entry, size_t ef, int level,
    size_t* comps, bool skip_deleted) const {
  VisitedPool& visited = scratch();
  visited.begin();

  // candidates is a min-heap of what to explore next, results is a max-heap of
  // the best ef found so far. the search stops when the nearest unexplored
  // candidate is further than the worst result, because the graph is a
  // proximity graph and going further out is very unlikely to improve things.
  std::priority_queue<std::pair<float, id_t>, std::vector<std::pair<float, id_t>>,
                      std::greater<>> candidates;
  std::priority_queue<std::pair<float, id_t>> results;

  const float d0 = dist(entry, q);
  if (comps) ++*comps;
  visited.test_and_set(entry);
  candidates.emplace(d0, entry);
  if (!skip_deleted || !tombstone_[entry]) results.emplace(d0, entry);

  while (!candidates.empty()) {
    auto [cd, cid] = candidates.top();
    if (results.size() >= ef && cd > results.top().first) break;
    candidates.pop();

    const auto& node_links = links_[cid];
    if (level >= static_cast<int>(node_links.size())) continue;
    const auto& neighbours = node_links[level];

    for (id_t nid : neighbours) {
      if (visited.test_and_set(nid)) continue;
      const float nd = dist(nid, q);
      if (comps) ++*comps;

      const bool keep = !skip_deleted || !tombstone_[nid];
      if (results.size() < ef || nd < results.top().first) {
        candidates.emplace(nd, nid);
        if (keep) {
          results.emplace(nd, nid);
          if (results.size() > ef) results.pop();
        }
      }
    }
  }
  return results;
}

// ---------------------------------------------------------- select_neighbours

void HnswIndex::select_neighbours(std::priority_queue<std::pair<float, id_t>>& candidates,
                                  size_t M, std::vector<id_t>& out) const {
  out.clear();
  if (candidates.size() <= M) {
    out.reserve(candidates.size());
    while (!candidates.empty()) {
      out.push_back(candidates.top().second);
      candidates.pop();
    }
    return;
  }

  // flip to nearest-first
  std::vector<std::pair<float, id_t>> work;
  work.reserve(candidates.size());
  while (!candidates.empty()) {
    work.push_back(candidates.top());
    candidates.pop();
  }
  std::reverse(work.begin(), work.end());

  // Algorithm 4 from the paper. Take a candidate only if it is closer to the
  // query node than to every neighbour already taken. Rejecting a candidate
  // that sits "behind" one we already have is what keeps the long range edges
  // that make the graph navigable, instead of M copies of the same direction.
  out.reserve(M);
  for (const auto& [cd, cid] : work) {
    if (out.size() >= M) break;
    bool dominated = false;
    const float* cvec = vector_at(cid);
    for (id_t kept : out) {
      if (distance(cvec, vector_at(kept), dim_, params_.metric) < cd) {
        dominated = true;
        break;
      }
    }
    if (!dominated) out.push_back(cid);
  }

  // The heuristic can return fewer than M. Backfill with the nearest rejects
  // so degree doesn't collapse on dense clusters and split the graph.
  if (out.size() < M) {
    for (const auto& [cd, cid] : work) {
      if (out.size() >= M) break;
      if (std::find(out.begin(), out.end(), cid) == out.end()) out.push_back(cid);
    }
  }
}

// ---------------------------------------------------------------- insert

void HnswIndex::insert_locked(const float* vec, label_t label) {
  const id_t id = static_cast<id_t>(count_);

  data_.resize((count_ + 1) * dim_);
  std::memcpy(data_.data() + size_t(id) * dim_, vec, dim_ * sizeof(float));
  if (params_.metric == Metric::Cosine) normalize(data_.data() + size_t(id) * dim_, dim_);

  const int level = random_level();
  levels_.push_back(level);
  labels_.push_back(label);
  tombstone_.push_back(0);
  links_.emplace_back(level + 1);
  by_label_[label] = id;
  ++count_;

  const float* q = vector_at(id);

  if (entry_ == INVALID_ID) {
    entry_ = id;
    max_level_ = level;
    return;
  }

  id_t cur = entry_;
  size_t comps = 0;

  // Phase 1: from the top down to level+1, plain greedy. One neighbour at a
  // time, no beam, because we only need a good entry point for the next phase.
  for (int lc = max_level_; lc > level; --lc) {
    bool improved = true;
    while (improved) {
      improved = false;
      if (lc >= static_cast<int>(links_[cur].size())) break;
      float best = dist(cur, q);
      for (id_t nid : links_[cur][lc]) {
        const float d = dist(nid, q);
        ++comps;
        if (d < best) {
          best = d;
          cur = nid;
          improved = true;
        }
      }
    }
  }

  // Phase 2: from min(level, max_level) down to 0, beam search and link up.
  for (int lc = std::min(level, max_level_); lc >= 0; --lc) {
    auto candidates = search_layer(q, cur, ef_construction_, lc, &comps,
                                   /*skip_deleted=*/false);
    if (candidates.empty()) continue;
    cur = candidates.top().second;

    const size_t Mmax = (lc == 0) ? maxM0_ : M_;
    std::vector<id_t> chosen;
    select_neighbours(candidates, M_, chosen);
    links_[id][lc] = chosen;

    // Edges are bidirectional. Adding the back-link can push a neighbour over
    // its cap, and re-running the heuristic there is what stops hubs forming.
    for (id_t nid : chosen) {
      if (lc >= static_cast<int>(links_[nid].size())) continue;
      auto& nl = links_[nid][lc];
      if (std::find(nl.begin(), nl.end(), id) != nl.end()) continue;

      if (nl.size() < Mmax) {
        nl.push_back(id);
        continue;
      }

      std::priority_queue<std::pair<float, id_t>> pool;
      const float* nvec = vector_at(nid);
      pool.emplace(distance(nvec, q, dim_, params_.metric), id);
      for (id_t existing : nl) {
        pool.emplace(distance(nvec, vector_at(existing), dim_, params_.metric), existing);
      }
      std::vector<id_t> pruned;
      select_neighbours(pool, Mmax, pruned);
      nl = pruned;
    }
  }

  if (level > max_level_) {
    max_level_ = level;
    entry_ = id;
  }
}

// ---------------------------------------------------------------- search

SearchResult HnswIndex::search_locked(const float* query, size_t k, size_t ef) const {
  if (entry_ == INVALID_ID || size() == 0) return {};

  // Temporary vector for cosine, since the index stores normalized rows and
  // the caller's query has no reason to be normalized already.
  std::vector<float> qbuf;
  const float* q = query;
  if (params_.metric == Metric::Cosine) {
    qbuf.assign(query, query + dim_);
    normalize(qbuf.data(), dim_);
    q = qbuf.data();
  }

  size_t comps = 0;
  id_t cur = entry_;

  for (int lc = max_level_; lc > 0; --lc) {
    bool improved = true;
    while (improved) {
      improved = false;
      if (lc >= static_cast<int>(links_[cur].size())) break;
      float best = dist(cur, q);
      for (id_t nid : links_[cur][lc]) {
        const float d = dist(nid, q);
        ++comps;
        if (d < best) {
          best = d;
          cur = nid;
          improved = true;
        }
      }
    }
  }

  // ef must be at least k or the beam can't hold the answer
  auto results = search_layer(q, cur, std::max(ef, k), 0, &comps, /*skip_deleted=*/true);
  while (results.size() > k) results.pop();
  return drain(results, comps);
}

// ---------------------------------------------------------------- stats

HnswIndex::Stats HnswIndex::stats() const {
  std::shared_lock lock(mutex_);
  size_t edges = 0, l0 = 0;
  for (id_t i = 0; i < count_; ++i) {
    for (const auto& lvl : links_[i]) edges += lvl.size();
    if (!links_[i].empty()) {
      l0 += links_[i][0].size();
    }
  }
  size_t mem = data_.capacity() * sizeof(float) + edges * sizeof(id_t) +
               count_ * (sizeof(int) + sizeof(label_t) + 1);
  const char* mname = params_.metric == Metric::L2 ? "l2"
                      : params_.metric == Metric::Cosine ? "cosine" : "ip";
  return Stats{size(),
               deleted_count_,
               capacity(),
               static_cast<size_t>(max_level_ + 1),
               dim_,
               edges,
               count_ ? double(l0) / double(count_) : 0.0,
               mem,
               mname,
               simd_backend()};
}

// ---------------------------------------------------------------- persistence

// Flat binary. Vectors are one contiguous block so the load is a single read
// rather than a per-node allocation, and the graph follows as varint-free
// fixed width ids, which keeps load simple and fast.
namespace {
constexpr char MAGIC[8] = {'A', 'N', 'N', 'L', 'I', 'T', 'E', '2'};

template <typename T>
void put(std::ostream& os, const T& v) {
  os.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

template <typename T>
T get(std::istream& is) {
  T v{};
  is.read(reinterpret_cast<char*>(&v), sizeof(T));
  return v;
}
}  // namespace

void HnswIndex::save(const std::string& path) const {
  std::shared_lock lock(mutex_);
  const std::string tmp = path + ".tmp";
  std::ofstream os(tmp, std::ios::binary);
  if (!os) throw std::runtime_error("cannot open " + tmp);

  os.write(MAGIC, sizeof(MAGIC));
  put<uint64_t>(os, dim_);
  put<uint64_t>(os, count_);
  put<uint64_t>(os, deleted_count_);
  put<uint64_t>(os, M_);
  put<uint64_t>(os, maxM0_);
  put<uint64_t>(os, ef_construction_);
  put<uint64_t>(os, ef_search_);
  put<uint64_t>(os, params_.seed);
  put<int32_t>(os, static_cast<int32_t>(params_.metric));
  put<int32_t>(os, max_level_);
  put<uint32_t>(os, entry_);

  os.write(reinterpret_cast<const char*>(data_.data()),
           std::streamsize(count_ * dim_ * sizeof(float)));

  for (id_t i = 0; i < count_; ++i) {
    put<int32_t>(os, levels_[i]);
    put<label_t>(os, labels_[i]);
    put<uint8_t>(os, tombstone_[i]);
    put<uint32_t>(os, static_cast<uint32_t>(links_[i].size()));
    for (const auto& lvl : links_[i]) {
      put<uint32_t>(os, static_cast<uint32_t>(lvl.size()));
      os.write(reinterpret_cast<const char*>(lvl.data()),
               std::streamsize(lvl.size() * sizeof(id_t)));
    }
  }
  os.flush();
  if (!os) throw std::runtime_error("write failed for " + tmp);
  os.close();
  // rename last so a crash mid-write can't leave a corrupt index behind
  if (std::rename(tmp.c_str(), path.c_str()) != 0) {
    throw std::runtime_error("rename failed for " + path);
  }
}

std::unique_ptr<HnswIndex> HnswIndex::load(const std::string& path) {
  std::ifstream is(path, std::ios::binary);
  if (!is) throw std::runtime_error("cannot open " + path);

  char magic[8];
  is.read(magic, sizeof(magic));
  if (std::memcmp(magic, MAGIC, sizeof(MAGIC)) != 0) {
    throw std::runtime_error("not an annlite index: " + path);
  }

  const auto dim = get<uint64_t>(is);
  const auto count = get<uint64_t>(is);
  const auto deleted = get<uint64_t>(is);
  const auto M = get<uint64_t>(is);
  const auto maxM0 = get<uint64_t>(is);
  const auto efc = get<uint64_t>(is);
  const auto efs = get<uint64_t>(is);
  const auto seed = get<uint64_t>(is);
  const auto metric = static_cast<Metric>(get<int32_t>(is));
  const auto max_level = get<int32_t>(is);
  const auto entry = get<uint32_t>(is);

  HnswParams p;
  p.M = M;
  p.ef_construction = efc;
  p.ef_search = efs;
  p.seed = seed;
  p.metric = metric;
  auto idx = std::make_unique<HnswIndex>(dim, p);
  idx->maxM0_ = maxM0;
  idx->count_ = count;
  idx->deleted_count_ = deleted;
  idx->max_level_ = max_level;
  idx->entry_ = entry;

  idx->data_.resize(count * dim);
  is.read(reinterpret_cast<char*>(idx->data_.data()),
          std::streamsize(count * dim * sizeof(float)));

  idx->levels_.resize(count);
  idx->labels_.resize(count);
  idx->tombstone_.resize(count);
  idx->links_.resize(count);
  for (uint64_t i = 0; i < count; ++i) {
    idx->levels_[i] = get<int32_t>(is);
    idx->labels_[i] = get<label_t>(is);
    idx->tombstone_[i] = get<uint8_t>(is);
    const auto nlevels = get<uint32_t>(is);
    idx->links_[i].resize(nlevels);
    for (uint32_t l = 0; l < nlevels; ++l) {
      const auto n = get<uint32_t>(is);
      idx->links_[i][l].resize(n);
      is.read(reinterpret_cast<char*>(idx->links_[i][l].data()),
              std::streamsize(n * sizeof(id_t)));
    }
    if (!idx->tombstone_[i]) idx->by_label_[idx->labels_[i]] = static_cast<id_t>(i);
  }
  if (!is) throw std::runtime_error("truncated index file: " + path);
  return idx;
}

}  // namespace annlite
