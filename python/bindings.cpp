// pybind11 bindings.
//
// numpy arrays go in and out without copying where possible. The GIL is
// released around every search and build, which is the whole reason a C++ core
// is worth having from Python: a thread pool of queries actually runs in
// parallel instead of taking turns.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <thread>
#include <vector>

#include "annlite/hnsw.hpp"

namespace py = pybind11;
using namespace annlite;

namespace {

Metric parse_metric(const std::string& s) {
  if (s == "l2" || s == "euclidean") return Metric::L2;
  if (s == "ip" || s == "inner_product" || s == "dot") return Metric::InnerProduct;
  if (s == "cosine") return Metric::Cosine;
  throw std::invalid_argument("metric must be one of l2, ip, cosine");
}

using farray = py::array_t<float, py::array::c_style | py::array::forcecast>;

void check_2d(const farray& a, size_t dim) {
  if (a.ndim() != 2) throw std::invalid_argument("expected a 2d array");
  if (static_cast<size_t>(a.shape(1)) != dim) {
    throw std::invalid_argument("dimension mismatch: index is " + std::to_string(dim) +
                                ", got " + std::to_string(a.shape(1)));
  }
}

}  // namespace

class PyIndex {
 public:
  PyIndex(size_t dim, const std::string& metric, size_t M, size_t ef_construction,
          size_t ef_search, uint64_t seed) {
    HnswParams p;
    p.M = M;
    p.ef_construction = ef_construction;
    p.ef_search = ef_search;
    p.seed = seed;
    p.metric = parse_metric(metric);
    idx_ = std::make_unique<HnswIndex>(dim, p);
  }

  explicit PyIndex(std::unique_ptr<HnswIndex> idx) : idx_(std::move(idx)) {}

  void add(farray vecs, std::vector<label_t> labels) {
    const size_t dim = idx_->dim();
    check_2d(vecs, dim);
    const size_t n = static_cast<size_t>(vecs.shape(0));
    if (labels.size() != n) {
      throw std::invalid_argument("got " + std::to_string(n) + " vectors but " +
                                  std::to_string(labels.size()) + " labels");
    }
    auto view = vecs.unchecked<2>();
    const float* base = view.data(0, 0);
    idx_->reserve(idx_->raw_size() + n);
    {
      py::gil_scoped_release release;
      for (size_t i = 0; i < n; ++i) idx_->add(base + i * dim, labels[i]);
    }
  }

  py::tuple search(farray queries, size_t k, size_t ef, int threads) {
    const size_t dim = idx_->dim();
    check_2d(queries, dim);
    const size_t n = static_cast<size_t>(queries.shape(0));
    auto view = queries.unchecked<2>();
    const float* base = view.data(0, 0);

    std::vector<label_t> labels(n * k, 0);
    std::vector<float> dists(n * k, 0.0f);
    std::vector<uint32_t> found(n, 0);
    size_t comps = 0;

    {
      py::gil_scoped_release release;
      auto run = [&](size_t lo, size_t hi, size_t* local) {
        for (size_t i = lo; i < hi; ++i) {
          auto r = idx_->search(base + i * dim, k, ef);
          found[i] = static_cast<uint32_t>(r.labels.size());
          for (size_t j = 0; j < r.labels.size(); ++j) {
            labels[i * k + j] = r.labels[j];
            dists[i * k + j] = r.distances[j];
          }
          *local += r.distance_computations;
        }
      };

      if (threads <= 1 || n < 2) {
        run(0, n, &comps);
      } else {
        const size_t nt = std::min<size_t>(threads, n);
        std::vector<std::thread> pool;
        std::vector<size_t> counts(nt, 0);
        const size_t chunk = (n + nt - 1) / nt;
        for (size_t t = 0; t < nt; ++t) {
          const size_t lo = t * chunk, hi = std::min(n, lo + chunk);
          if (lo >= hi) break;
          pool.emplace_back([&, lo, hi, t] { run(lo, hi, &counts[t]); });
        }
        for (auto& th : pool) th.join();
        for (size_t c : counts) comps += c;
      }
    }

    // shorter result rows get padded with -1 so the caller always gets a
    // rectangular array and can tell padding from a real label
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = found[i]; j < k; ++j) {
        labels[i * k + j] = static_cast<label_t>(-1);
        dists[i * k + j] = std::numeric_limits<float>::infinity();
      }
    }

    py::array_t<label_t> out_labels({n, k});
    py::array_t<float> out_dists({n, k});
    std::memcpy(out_labels.mutable_data(), labels.data(), labels.size() * sizeof(label_t));
    std::memcpy(out_dists.mutable_data(), dists.data(), dists.size() * sizeof(float));
    return py::make_tuple(out_labels, out_dists, comps);
  }

  py::tuple brute_force(farray queries, size_t k) {
    const size_t dim = idx_->dim();
    check_2d(queries, dim);
    const size_t n = static_cast<size_t>(queries.shape(0));
    auto view = queries.unchecked<2>();
    const float* base = view.data(0, 0);

    std::vector<label_t> labels(n * k, static_cast<label_t>(-1));
    std::vector<float> dists(n * k, std::numeric_limits<float>::infinity());
    {
      py::gil_scoped_release release;
      for (size_t i = 0; i < n; ++i) {
        auto r = idx_->brute_force(base + i * dim, k);
        for (size_t j = 0; j < r.labels.size(); ++j) {
          labels[i * k + j] = r.labels[j];
          dists[i * k + j] = r.distances[j];
        }
      }
    }
    py::array_t<label_t> out_labels({n, k});
    py::array_t<float> out_dists({n, k});
    std::memcpy(out_labels.mutable_data(), labels.data(), labels.size() * sizeof(label_t));
    std::memcpy(out_dists.mutable_data(), dists.data(), dists.size() * sizeof(float));
    return py::make_tuple(out_labels, out_dists);
  }

  bool remove(label_t label) { return idx_->remove(label); }
  bool contains(label_t label) const { return idx_->contains(label); }
  size_t size() const { return idx_->size(); }
  size_t dim() const { return idx_->dim(); }
  size_t deleted() const { return idx_->deleted(); }
  size_t ef_search() const { return idx_->ef_search(); }
  void set_ef_search(size_t ef) { idx_->set_ef_search(ef); }

  std::vector<float> get_vector(label_t label) const { return idx_->get_vector(label); }

  void save(const std::string& path) const {
    py::gil_scoped_release release;
    idx_->save(path);
  }

  static PyIndex* load(const std::string& path) {
    std::unique_ptr<HnswIndex> idx;
    {
      py::gil_scoped_release release;
      idx = HnswIndex::load(path);
    }
    return new PyIndex(std::move(idx));
  }

  py::dict stats() const {
    auto s = idx_->stats();
    py::dict d;
    d["elements"] = s.elements;
    d["deleted"] = s.deleted;
    d["capacity"] = s.capacity;
    d["levels"] = s.max_level;
    d["dim"] = s.dim;
    d["total_edges"] = s.total_edges;
    d["mean_degree_layer0"] = s.mean_degree_layer0;
    d["memory_bytes"] = s.memory_bytes;
    d["metric"] = s.metric;
    d["simd"] = s.simd;
    return d;
  }

 private:
  std::unique_ptr<HnswIndex> idx_;
};

PYBIND11_MODULE(_annlite, m) {
  m.doc() = "HNSW vector index, C++ core";
  m.attr("__version__") = "0.1.0";
  m.def("simd_backend", &simd_backend, "which distance kernel was compiled in");

  py::class_<PyIndex>(m, "Index")
      .def(py::init<size_t, const std::string&, size_t, size_t, size_t, uint64_t>(),
           py::arg("dim"), py::arg("metric") = "l2", py::arg("M") = 16,
           py::arg("ef_construction") = 200, py::arg("ef_search") = 50,
           py::arg("seed") = 100)
      .def("add", &PyIndex::add, py::arg("vectors"), py::arg("labels"))
      .def("search", &PyIndex::search, py::arg("queries"), py::arg("k") = 10,
           py::arg("ef") = 0, py::arg("threads") = 1)
      .def("brute_force", &PyIndex::brute_force, py::arg("queries"), py::arg("k") = 10)
      .def("remove", &PyIndex::remove, py::arg("label"))
      .def("contains", &PyIndex::contains, py::arg("label"))
      .def("get_vector", &PyIndex::get_vector, py::arg("label"))
      .def("save", &PyIndex::save, py::arg("path"))
      .def_static("load", &PyIndex::load, py::arg("path"))
      .def("stats", &PyIndex::stats)
      .def_property("ef_search", &PyIndex::ef_search, &PyIndex::set_ef_search)
      .def_property_readonly("size", &PyIndex::size)
      .def_property_readonly("dim", &PyIndex::dim)
      .def_property_readonly("deleted", &PyIndex::deleted)
      .def("__len__", &PyIndex::size)
      .def("__contains__", &PyIndex::contains);
}
