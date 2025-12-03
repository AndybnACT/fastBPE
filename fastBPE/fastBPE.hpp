#pragma once

#include <algorithm>
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <list>
#include <set>
#include <stdio.h>
#include <string>
#include <cstring>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h> // ftruncate
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <omp.h>
#include <chrono>
#include <cmath>
#ifdef CONFIG_OMP_TBB
#include <tbb/concurrent_unordered_map.h>
#define _hash_map tbb::concurrent_unordered_map
#else
#define _hash_map unordered_map
#endif

#ifdef CONFIG_MPI
#include <boost/mpi.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/unordered_map.hpp>
namespace mpi = boost::mpi;
#endif

#include <cstdint>
#include <vector>
#include <cstring>

#ifdef CONFIG_MPI
#include <mpi.h>
#endif

#ifdef CONFIG_MPI
static void serialize_map(
    const std::unordered_map<std::string, std::string> &m,
    std::vector<char> &buffer)
{
    buffer.clear();
    buffer.reserve(m.size() * 32); 

    std::uint64_t n_pairs = static_cast<std::uint64_t>(m.size());
    buffer.insert(buffer.end(),
                  reinterpret_cast<const char *>(&n_pairs),
                  reinterpret_cast<const char *>(&n_pairs) + sizeof(std::uint64_t));

    for (const auto &kv : m) {
        const std::string &k = kv.first;
        const std::string &v = kv.second;

        std::uint32_t klen = static_cast<std::uint32_t>(k.size());
        std::uint32_t vlen = static_cast<std::uint32_t>(v.size());

        buffer.insert(buffer.end(),
                      reinterpret_cast<const char *>(&klen),
                      reinterpret_cast<const char *>(&klen) + sizeof(std::uint32_t));
        buffer.insert(buffer.end(),
                      reinterpret_cast<const char *>(&vlen),
                      reinterpret_cast<const char *>(&vlen) + sizeof(std::uint32_t));

        buffer.insert(buffer.end(), k.data(), k.data() + k.size());
        buffer.insert(buffer.end(), v.data(), v.data() + v.size());
    }
}

#ifdef CONFIG_MPI
#include <mpi.h>
#include <cstdint>
#include <vector>
#include <cstring>

// word_count: unordered_map<string, uint32_t>
static void serialize_word_count(
    const std::unordered_map<std::string, uint32_t> &m,
    std::vector<char> &buffer)
{
    buffer.clear();
    buffer.reserve(m.size() * 16); 

    std::uint64_t n_pairs = static_cast<std::uint64_t>(m.size());
    buffer.insert(buffer.end(),
                  reinterpret_cast<const char *>(&n_pairs),
                  reinterpret_cast<const char *>(&n_pairs) + sizeof(std::uint64_t));

    for (const auto &kv : m) {
        const std::string &k = kv.first;
        std::uint32_t v      = kv.second;

        std::uint32_t klen = static_cast<std::uint32_t>(k.size());

        buffer.insert(buffer.end(),
                      reinterpret_cast<const char *>(&klen),
                      reinterpret_cast<const char *>(&klen) + sizeof(std::uint32_t));
        buffer.insert(buffer.end(),
                      reinterpret_cast<const char *>(&v),
                      reinterpret_cast<const char *>(&v) + sizeof(std::uint32_t));
        buffer.insert(buffer.end(), k.data(), k.data() + k.size());
    }
}

static void deserialize_word_count(
    const char *data, std::size_t len,
    std::unordered_map<std::string, uint32_t> &out)
{
    using namespace std;

    const char *p   = data;
    const char *end = data + len;

    if (end - p < static_cast<std::ptrdiff_t>(sizeof(std::uint64_t))) {
        return;
    }

    std::uint64_t n_pairs = 0;
    std::memcpy(&n_pairs, p, sizeof(std::uint64_t));
    p += sizeof(std::uint64_t);

    out.clear();
    out.reserve(static_cast<size_t>(n_pairs));

    for (std::uint64_t i = 0; i < n_pairs && p < end; ++i) {
        if (end - p < static_cast<std::ptrdiff_t>(2 * sizeof(std::uint32_t))) {
            break;
        }

        std::uint32_t klen = 0, v = 0;
        std::memcpy(&klen, p, sizeof(std::uint32_t)); p += sizeof(std::uint32_t);
        std::memcpy(&v,   p, sizeof(std::uint32_t)); p += sizeof(std::uint32_t);

        if (end - p < static_cast<std::ptrdiff_t>(klen)) {
            break;
        }

        std::string key(p, p + klen);
        p += klen;

        out.emplace(std::move(key), v);
    }
}
#endif // CONFIG_MPI

static void deserialize_map_segment(
    const char *data, std::size_t len,
    std::unordered_map<std::string, std::string> &out)
{
    const char *p   = data;
    const char *end = data + len;

    if (end - p < static_cast<std::ptrdiff_t>(sizeof(std::uint64_t))) {
        return;
    }

    std::uint64_t n_pairs = 0;
    std::memcpy(&n_pairs, p, sizeof(std::uint64_t));
    p += sizeof(std::uint64_t);

    for (std::uint64_t i = 0; i < n_pairs && p < end; ++i) {
        if (end - p < static_cast<std::ptrdiff_t>(2 * sizeof(std::uint32_t))) {
            break;
        }
        std::uint32_t klen = 0, vlen = 0;
        std::memcpy(&klen, p, sizeof(std::uint32_t)); p += sizeof(std::uint32_t);
        std::memcpy(&vlen, p, sizeof(std::uint32_t)); p += sizeof(std::uint32_t);

        if (end - p < static_cast<std::ptrdiff_t>(klen + vlen)) {
            break;
        }

        std::string key(p, p + klen);
        p += klen;

        std::string val(p, p + vlen);
        p += vlen;

        out.emplace(std::move(key), std::move(val));
    }
}
#endif // CONFIG_MPI

namespace fastBPE {

using namespace std;

const size_t kMaxPairs = 1000 * 1000 * 1000;
const size_t kThreads = max(1, min(10, int(thread::hardware_concurrency())));
const char *kEndWord = "</w>";
const size_t kEndWordLength = 4;
const char *kTokenDelim = "@@";
const size_t kTokenDelimLength = 2;

int safeOpen(const char *file_path, int flags, mode_t mode = 0) {
  int fd = open(file_path, flags, mode);
  if (fd < 0) {
    fprintf(stderr, "Cannot open text file %s\n", file_path);
    exit(EXIT_FAILURE);
  }
  return fd;
}

vector<size_t> get_boundary(char *f, size_t size, size_t nr_threads)
{
  vector<size_t> boundary(nr_threads + 1);

  boundary[0] = 0;
  boundary[nr_threads] = size;
  for (size_t i = 1; i < nr_threads; i++) {
    size_t start = (size / nr_threads) * i;
    while (f[start] != ' ' && f[start] != '\n') {
      start++;
      if (start >= size) {
        fprintf(stderr, "error dividing works for output for #%zu, reaching the end\n", i);
	start = size;
	break;
      }
    }
    boundary[i] = start + 1 >= size ? size : start + 1;
  }
  return boundary;
}

size_t readText(const char *fp, _hash_map<std::string, uint32_t> &word_count) {
  using namespace std;

  word_count.clear();
  string cur;
  uint64_t total = 0;
  size_t sz = 0;

  auto handle_char = [] (char cur_char,
                         std::string &cur_word,
                         _hash_map<std::string, uint32_t> &wc,
                         uint64_t &tot) {
    if (cur_char == ' ' || cur_char == '\n') {
      if (cur_word.empty())
        return;
      wc[cur_word] += 1;
      ++tot;
      cur_word.clear();
    } else {
      cur_word.push_back(cur_char);
    }
  };

  if (string(fp) == "-") {
    for (std::string line; std::getline(std::cin, line);) {
      for (char c : line) {
        handle_char(c, cur, word_count, total);
      }
      handle_char('\n', cur, word_count, total);
    }
    return 0;  
  }

  int fd = safeOpen(fp, O_RDONLY);

  struct stat s;
  if (fstat(fd, &s) < 0) {
    fprintf(stderr, "fstat failed on %s\n", fp);
    exit(EXIT_FAILURE);
  }

  size_t size = static_cast<size_t>(s.st_size);
  sz = size;

  char *f = (char *)mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (f == MAP_FAILED) {
    fprintf(stderr, "Input memory map failed : %d.\n", errno);
    close(fd);
    exit(EXIT_FAILURE);
  }

#ifdef CONFIG_OMP
  int nr_threads = 1;
  #pragma omp parallel
  {
    if (omp_get_thread_num() == 0) {
      nr_threads = omp_get_num_threads();
    }
  }

  std::vector<size_t> boundary = get_boundary(f, size, nr_threads);

  std::vector<std::unordered_map<std::string, uint32_t>> local_wc(nr_threads);
  std::vector<uint64_t> local_tot(nr_threads, 0);

  #pragma omp parallel for schedule(static)
  for (int t = 0; t < nr_threads; ++t) {
    std::string cur_word;
    auto &wc  = local_wc[t];
    auto &tot = local_tot[t];

    size_t begin = boundary[t];
    size_t end   = boundary[t + 1];

    for (size_t i = begin; i < end; ++i) {
      char c = f[i];
      handle_char(c, cur_word, wc, tot);
    }

    if (!cur_word.empty()) {
      wc[cur_word] += 1;
      ++tot;
      cur_word.clear();
    }
  }

  for (int t = 0; t < nr_threads; ++t) {
    for (auto &kv : local_wc[t]) {
      word_count[kv.first] += kv.second;
    }
    total += local_tot[t];
  }

#else
  for (size_t i = 0; i < size; ++i) {
    char c = f[i];
    handle_char(c, cur, word_count, total);
  }
  if (!cur.empty()) {
    word_count[cur] += 1;
    ++total;
    cur.clear();
  }
#endif

  munmap(f, size);
  close(fd);

  return sz;
}

#ifdef CONFIG_MPI
size_t readText_mpi_parallel(const char *fp,
                             std::unordered_map<std::string, uint32_t> &word_count,
                             MPI_Comm comm)
{
    using namespace std;

    int rank = 0, nprocs = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    if (std::string(fp) == "-") {
        if (rank == 0) {
            fprintf(stderr, "readText_mpi_parallel: stdin ('-') not supported in MPI mode.\n");
        }
        MPI_Abort(comm, 1);
    }

    size_t filesize = 0;
    std::vector<size_t> boundaries;

    // ===== rank 0: get the file size + use mmap + get_boundary to get MPI boundary =====
    if (rank == 0) {
        int fd = safeOpen(fp, O_RDONLY);

        struct stat s;
        if (fstat(fd, &s) < 0) {
            fprintf(stderr, "fstat failed on %s\n", fp);
            exit(EXIT_FAILURE);
        }

        filesize = static_cast<size_t>(s.st_size);

        char *f = (char *)mmap(NULL, filesize, PROT_READ, MAP_PRIVATE, fd, 0);
        if (f == MAP_FAILED) {
            fprintf(stderr, "Input memory map failed on %s : %d\n", fp, errno);
            close(fd);
            exit(EXIT_FAILURE);
        }

        boundaries = get_boundary(f, filesize, nprocs);

        munmap(f, filesize);
        close(fd);
    }

    // filesize
    MPI_Bcast(&filesize, 1, MPI_UNSIGNED_LONG, 0, comm);

    // boundaries[nprocs+1]
    if (rank != 0) {
        boundaries.resize(nprocs + 1);
    }
    MPI_Bcast(boundaries.data(),
              static_cast<int>(nprocs + 1),
              MPI_UNSIGNED_LONG,
              0, comm);


    size_t begin = boundaries[rank];
    size_t end   = boundaries[rank + 1];
    if (begin > end) begin = end;
    size_t chunk_size = end - begin;

    // =====  rank：pread its own chunk  =====
    std::unordered_map<std::string, uint32_t> local_wc;
    uint64_t local_total = 0;

    auto handle_char = [] (char cur_char,
                           std::string &cur_word,
                           std::unordered_map<std::string, uint32_t> &wc,
                           uint64_t &tot) {
        if (cur_char == ' ' || cur_char == '\n') {
            if (cur_word.empty())
                return;
            wc[cur_word] += 1;
            ++tot;
            cur_word.clear();
        } else {
            cur_word.push_back(cur_char);
        }
    };

    if (chunk_size > 0) {
        int fd = safeOpen(fp, O_RDONLY);

        std::vector<char> buf(chunk_size);
        size_t to_read  = chunk_size;
        size_t offset   = 0;
        while (to_read > 0) {
            ssize_t n = pread(fd, buf.data() + offset, to_read,
                              static_cast<off_t>(begin + offset));
            if (n < 0) {
                fprintf(stderr, "pread failed on %s : %d\n", fp, errno);
                close(fd);
                MPI_Abort(comm, 1);
            }
            if (n == 0) {
                // unexpected EOF
                break;
            }
            offset  += static_cast<size_t>(n);
            to_read -= static_cast<size_t>(n);
        }

        close(fd);

        std::string cur_word;
        for (size_t i = 0; i < offset; ++i) {
            char c = buf[i];
            handle_char(c, cur_word, local_wc, local_total);
        }

        if (!cur_word.empty()) {
            local_wc[cur_word] += 1;
            ++local_total;
            cur_word.clear();
        }
    }

    // ===== rank 0 收集并合并所有 rank 的 local_wc =====
    std::vector<char> sendbuf;
    serialize_word_count(local_wc, sendbuf);
    std::uint64_t send_size = static_cast<std::uint64_t>(sendbuf.size());

    std::vector<std::uint64_t> recv_sizes;
    if (rank == 0) {
        recv_sizes.resize(nprocs);
    }

    MPI_Gather(&send_size, 1, MPI_UINT64_T,
               rank == 0 ? recv_sizes.data() : nullptr, 1, MPI_UINT64_T,
               0, comm);

    std::vector<int> recvcounts;
    std::vector<int> displs;
    std::vector<char> recvbuf;

    if (rank == 0) {
        recvcounts.resize(nprocs);
        displs.resize(nprocs);

        std::size_t total_bytes = 0;
        for (int i = 0; i < nprocs; ++i) {
            recvcounts[i] = static_cast<int>(recv_sizes[i]);
            displs[i]     = static_cast<int>(total_bytes);
            total_bytes  += recv_sizes[i];
        }
        recvbuf.resize(total_bytes);
    }

    MPI_Gatherv(sendbuf.data(),
                static_cast<int>(send_size),
                MPI_BYTE,
                rank == 0 ? recvbuf.data()   : nullptr,
                rank == 0 ? recvcounts.data(): nullptr,
                rank == 0 ? displs.data()    : nullptr,
                MPI_BYTE,
                0, comm);

    if (rank == 0) {
        word_count.clear();
        std::unordered_map<std::string, uint32_t> tmp;

        for (int i = 0; i < nprocs; ++i) {
            const char *ptr = recvbuf.data() + displs[i];
            std::size_t len = static_cast<std::size_t>(recv_sizes[i]);

            deserialize_word_count(ptr, len, tmp);
            for (auto &kv : tmp) {
                word_count[kv.first] += kv.second;
            }
        }
    }

    // 所有 rank 返回同样的 filesize（rank != 0 的 word_count 暂时没用）
    return filesize;
}
#endif // CONFIG_MPI



std::pair<size_t, uint64_t> output_or_count(
    const std::unordered_map<std::string, std::string> &bpe,
    size_t size,
    const char *f,
    char *fo)
{
    using namespace std;

    const char *end = f + size;
    string cur_word;
    cur_word.reserve(64); 

    size_t   charOut = 0;
    uint64_t total   = 0;

    auto flush_word = [&](char sep) {
        if (cur_word.empty()) {
            if (fo) fo[charOut] = sep;
            ++charOut;
            return;
        }

        auto it = bpe.find(cur_word);
        if (it == bpe.end()) {
            if (fo) {
                memcpy(fo + charOut, cur_word.data(), cur_word.size());
            }
            charOut += cur_word.size();
        } else {
            const std::string &out = it->second;
            if (fo) {
                memcpy(fo + charOut, out.data(), out.size());
            }
            charOut += out.size();
        }

        if (fo) fo[charOut] = sep;
        ++charOut;

        ++total;
        cur_word.clear();
    };

    for (const char *p = f; p != end; ++p) {
        char c = *p;
        if (c == ' ' || c == '\n') {
            flush_word(c);
        } else {
            cur_word.push_back(c);
        }
    }

    if (!cur_word.empty()) {
        auto it = bpe.find(cur_word);
        if (it == bpe.end()) {
            if (fo) {
                memcpy(fo + charOut, cur_word.data(), cur_word.size());
            }
            charOut += cur_word.size();
        } else {
            const std::string &out = it->second;
            if (fo) {
                memcpy(fo + charOut, out.data(), out.size());
            }
            charOut += out.size();
        }
        ++total;
    }

    return {charOut, total};
}


void outputText(const char *fpo, const char *fp,
                std::unordered_map<std::string, std::string> &bpe) {

  using namespace std;

  int fd = safeOpen(fp, O_RDONLY);
  int fdOut = safeOpen(fpo, O_WRONLY | O_CREAT | O_TRUNC, 0666);

  struct stat s;
  if (fstat(fd, &s) < 0) {
    fprintf(stderr, "fstat failed on %s\n", fp);
    exit(EXIT_FAILURE);
  }

  fprintf(stderr, "Applying BPE to %s ...\n", fp);
  size_t size = static_cast<size_t>(s.st_size);

  char *f = (char *)mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (f == MAP_FAILED) {
    fprintf(stderr, "Input memory map failed : %d.\n", errno);
    exit(EXIT_FAILURE);
  }

  auto p = output_or_count(bpe, size, f, nullptr);
  size_t out_size = p.first;

  char *fo = (char *)malloc(out_size);
  if (!fo) {
    fprintf(stderr, "Failed to allocate %zu bytes for output buffer.\n", out_size);
    munmap(f, size);
    close(fdOut);
    close(fd);
    exit(EXIT_FAILURE);
  }

  p = output_or_count(bpe, size, f, fo);

  size_t written = 0;
  while (written < out_size) {
    ssize_t n = write(fdOut, fo + written, out_size - written);
    if (n < 0) {
      fprintf(stderr, "Error writing to output file %s: errno=%d\n", fpo, errno);
      free(fo);
      munmap(f, size);
      close(fdOut);
      close(fd);
      exit(EXIT_FAILURE);
    }
    written += static_cast<size_t>(n);
  }

  fprintf(stderr, "Modified %lu words from text file.\n",
          static_cast<unsigned long>(p.second));

  free(fo);
  munmap(f, size);
  close(fdOut);
  close(fd);
}


struct pair_hash {
  template <class T1, class T2> size_t operator()(const pair<T1, T2> &p) const {
    auto h1 = hash<T1>{}(p.first);
    auto h2 = hash<T2>{}(p.second);
    size_t seed = h1;
    // boost::hash_combine
    return h2 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }
};

void tokenize(const unordered_map<string, uint32_t> &word_count,
              unordered_map<string, uint32_t> &token_to_int,
              vector<string> &int_to_token, vector<list<uint32_t>> &words,
              vector<int32_t> &counts) {

  for (auto &x : word_count) {
    auto &word = x.first;

    words.push_back(list<uint32_t>());
    auto &current_word = words.back();
    counts.push_back(x.second);

    int pos = 0, realLength = 0;
    int lastStart = 0;
    while (word[pos]) {
      bool newChar = (word[pos] & 0xc0) != 0x80; // not a continuation byte
      realLength += newChar;
      // new token
      if (newChar && pos > 0) {
        auto new_token = word.substr(lastStart, pos - lastStart);
        if (token_to_int.count(new_token) == 0) {
          int_to_token.push_back(new_token);
          token_to_int[new_token] = int_to_token.size() - 1;
        }
        current_word.push_back(token_to_int[new_token]);
        lastStart = pos;
      }
      pos++;
    }
    auto new_token = word.substr(lastStart, string::npos) + kEndWord;
    if (token_to_int.count(new_token) == 0) {
      int_to_token.push_back(new_token);
      token_to_int[new_token] = int_to_token.size() - 1;
    }
    current_word.push_back(token_to_int[new_token]);
  }
}

void tokenize_str(const unordered_map<string, uint32_t> &word_count,
                  unordered_map<string, vector<string>> &words) {

  for (auto &x : word_count) {
    auto &word = x.first;
    words[word] = vector<string>();

    int pos = 0, realLength = 0;
    int lastStart = 0;
    while (word[pos]) {
      bool newChar = (word[pos] & 0xc0) != 0x80; // not a continuation byte
      realLength += newChar;
      // new token
      if (newChar && pos > 0) {
        auto new_token = word.substr(lastStart, pos - lastStart);
        words[word].push_back(new_token);
        lastStart = pos;
      }
      pos++;
    }
    auto new_token = word.substr(lastStart, string::npos) + kEndWord;
    words[word].push_back(new_token);
  }
}

using tp = pair<uint32_t, uint32_t>;
using tps = pair<string, string>;
using pc = unordered_map<tp, pair<int32_t, tp> *, pair_hash>;

void count_in_word(
    list<uint32_t> &word, uint32_t wi, uint32_t count, pc &pair_counts,
    vector<pair<int32_t, tp>> &contiguous_counts,
    unordered_map<tp, unordered_set<uint32_t>, pair_hash> &where) {
  bool second = false;
  tp cur_pair;
  for (uint32_t token : word) {
    if (second) {
      cur_pair.first = cur_pair.second;
    }
    cur_pair.second = token;
    if (second) {
      auto it = pair_counts.find(cur_pair);
      if (it == pair_counts.end()) {
        contiguous_counts.emplace_back(0, cur_pair);
        auto *added = &contiguous_counts.back();
        pair_counts.emplace(piecewise_construct, forward_as_tuple(cur_pair),
                            forward_as_tuple(added));
        where[cur_pair].emplace();
      }
      if (count > 0) {
        where[cur_pair].insert(wi);
      } else {
        where[cur_pair].erase(wi);
      }
      pair_counts[cur_pair]->first += count;
    } else {
      second = true;
    }
  }
}

void find_maxp(vector<pair<int32_t, tp>> &contiguous_counts, tp &maxp,
               int32_t &max_c) {
  max_c = 0;
  for (auto &x : contiguous_counts) {
    if (x.first > max_c) {
      max_c = x.first;
      maxp = x.second;
    } else if (x.first == max_c and x.second < maxp) {
      maxp = x.second;
    }
  }
}

void getvocab(const char *inputFile1, const char *inputFile2) {
  // get vocab
  unordered_map<string, uint32_t> word_count;
  readText(inputFile1, word_count);
  if (strcmp(inputFile2, "") != 0) {
    readText(inputFile2, word_count);
  }

  // sort vocab
  auto compFunctor = [](pair<string, int> elem1, pair<string, int> elem2) {
    return elem1.second > elem2.second ||
           (elem1.second == elem2.second && elem1.first < elem2.first);
  };
  set<pair<string, int>, decltype(compFunctor)> sorted_vocab(
      word_count.begin(), word_count.end(), compFunctor);
  assert(word_count.size() == sorted_vocab.size());

  // print sorted vocab
  for (auto element : sorted_vocab)
    cout << element.first << " " << element.second << endl;
}

void learnbpe(const uint32_t kNPairs, const char *inputFile1,
              const char *inputFile2) {
  // get vocab
  unordered_map<string, uint32_t> word_count;
  readText(inputFile1, word_count);
  if (strcmp(inputFile2, "") != 0) {
    readText(inputFile2, word_count);
  }

  // a token is an int, it represents a string
  unordered_map<string, uint32_t> token_to_int;
  vector<string> int_to_token;

  vector<list<uint32_t>> words;
  vector<int32_t> counts;

  tokenize(word_count, token_to_int, int_to_token, words, counts);

  vector<pair<int32_t, tp>> contiguous_counts;
  contiguous_counts.reserve(kMaxPairs);

  pc pair_counts;
  unordered_map<tp, unordered_set<uint32_t>, pair_hash> where_to_update;

  tp cur_pair;
  int32_t max_c = 0;
  tp max_p;
  for (uint32_t wi = 0; wi < words.size(); wi++) {
    count_in_word(words[wi], wi, counts[wi], pair_counts, contiguous_counts,
                  where_to_update);
  }
  find_maxp(contiguous_counts, max_p, max_c);
  for (size_t i = 0; i < kNPairs; i++) {
    // create new token for pair. replace
    auto new_token = int_to_token[max_p.first] + int_to_token[max_p.second];
    cout << int_to_token[max_p.first] << " " << int_to_token[max_p.second]
         << " " << max_c << endl;

    uint32_t new_token_id = int_to_token.size();
    int_to_token.push_back(new_token);
    token_to_int[new_token] = new_token_id;
    max_c = 0;
    auto change_count = [&](tp pair, int32_t v, uint32_t wi) {
      auto it = pair_counts.find(pair);
      if (it != pair_counts.end()) {
        // assert(it->second + v >= 0);
        it->second->first += v;
      } else {
        if (v > 0) {
          contiguous_counts.emplace_back(v, pair);
          pair_counts.emplace(piecewise_construct, forward_as_tuple(pair),
                              forward_as_tuple(&(contiguous_counts.back())));
          where_to_update[pair] = unordered_set<uint32_t>();
        }
      }
      if (v > 0)
        where_to_update[pair].insert(wi);
    };

    for (auto wi : where_to_update[max_p]) {
      auto &cur_word = words[wi];
      auto it = cur_word.begin();
      bool second = false;
      while (it != cur_word.end()) {
        if (second) {
          cur_pair.first = cur_pair.second;
        }
        cur_pair.second = *it;

        if (second) {
          // found the pair
          if (cur_pair == max_p) {
            it--; // points to first element of pair
            // if there is a token before us
            if (it != cur_word.begin()) {
              it--;
              change_count(make_pair(*it, cur_pair.first), -counts[wi], wi);
              change_count(make_pair(*it, new_token_id), counts[wi], wi);
              it++;
            }

            it = cur_word.insert(it, new_token_id); // it points to new token
            it++;                    // it points to first element of pair
            it = cur_word.erase(it); // it points to second element of pair
            it = cur_word.erase(it); // it points to next value

            // if there is a token after the one we inserted
            if (it != cur_word.end()) {
              change_count(make_pair(cur_pair.second, *it), -counts[wi], wi);
              change_count(make_pair(new_token_id, *it), counts[wi], wi);
            }
            cur_pair.second = new_token_id;
          } else {
            it++;
          }
        } else {
          second = true;
          it++;
        }
      }
    }

    if (pair_counts.find(max_p) != pair_counts.end()){
      pair_counts[max_p]->first = 0;
    }
    find_maxp(contiguous_counts, max_p, max_c);
  }
}

void split(vector<string> &splits, const string &text, char sep) {
  size_t start = 0, end = 0;
  while ((end = text.find(sep, start)) != string::npos) {
    if (end != start)
      splits.push_back(text.substr(start, end - start));
    start = end + 1;
  }
  if (end != start && start < text.size())
    splits.push_back(text.substr(start));
}

string decode_gpt2_display_chars(const string& input) {
    // Create the reverse mapping table for common GPT-2 display characters
    // This maps Unicode codepoints to their original byte values
    static const std::unordered_map<uint32_t, uint8_t> unicode_to_byte = {
        // Space and control characters (0-32)
        {0x0100, 0},    // Ā -> null
        {0x0101, 1},    // ā -> SOH
        {0x0102, 2},    // Ă -> STX
        {0x0103, 3},    // ă -> ETX
        {0x0104, 4},    // Ą -> EOT
        {0x0105, 5},    // ą -> ENQ
        {0x0106, 6},    // Ć -> ACK
        {0x0107, 7},    // ć -> BEL
        {0x0108, 8},    // Ĉ -> BS
        {0x0109, 9},    // ĉ -> TAB
        {0x010A, 10},   // Ċ -> LF (newline)
        {0x010B, 11},   // ċ -> VT
        {0x010C, 12},   // Č -> FF
        {0x010D, 13},   // č -> CR (carriage return)
        {0x010E, 14},   // Ď -> SO
        {0x010F, 15},   // ď -> SI
        {0x0110, 16},   // Đ -> DLE
        {0x0111, 17},   // đ -> DC1
        {0x0112, 18},   // Ē -> DC2
        {0x0113, 19},   // ē -> DC3
        {0x0114, 20},   // Ĕ -> DC4
        {0x0115, 21},   // ĕ -> NAK
        {0x0116, 22},   // Ė -> SYN
        {0x0117, 23},   // ė -> ETB
        {0x0118, 24},   // Ę -> CAN
        {0x0119, 25},   // ę -> EM
        {0x011A, 26},   // Ě -> SUB
        {0x011B, 27},   // ě -> ESC
        {0x011C, 28},   // Ĝ -> FS
        {0x011D, 29},   // ĝ -> GS
        {0x011E, 30},   // Ğ -> RS
        {0x011F, 31},   // ğ -> US
        {0x0120, 32},   // Ġ -> space

        // Characters 127-160 (DEL and extended ASCII control)
        {0x0121, 127},  // ġ -> DEL
        {0x0122, 128},  // Ģ
        {0x0123, 129},  // ģ
        {0x0124, 130},  // Ĥ
        {0x0125, 131},  // ĥ
        {0x0126, 132},  // Ħ
        {0x0127, 133},  // ħ
        {0x0128, 134},  // Ĩ
        {0x0129, 135},  // ĩ
        {0x012A, 136},  // Ī
        {0x012B, 137},  // ī
        {0x012C, 138},  // Ĭ
        {0x012D, 139},  // ĭ
        {0x012E, 140},  // Į
        {0x012F, 141},  // į
        {0x0130, 142},  // İ
        {0x0131, 143},  // ı
        {0x0132, 144},  // Ĳ
        {0x0133, 145},  // ĳ
        {0x0134, 146},  // Ĵ
        {0x0135, 147},  // ĵ
        {0x0136, 148},  // Ķ
        {0x0137, 149},  // ķ
        {0x0138, 150},  // ĸ
        {0x0139, 151},  // Ĺ
        {0x013A, 152},  // ĺ
        {0x013B, 153},  // Ļ
        {0x013C, 154},  // ļ
        {0x013D, 155},  // Ľ
        {0x013E, 156},  // ľ
        {0x013F, 157},  // Ŀ
        {0x0140, 158},  // ŀ
        {0x0141, 159},  // Ł
        {0x0142, 160},  // ł

        // Characters 161-172 (¡ to ¬) are handled specially by GPT-2,
        // they're actually not remapped, but we skip them in the mapping
        // continuing at 173 (soft hyphen)
        {0x0143, 173},  // Ń -> soft hyphen
    };

    std::string result;
    result.reserve(input.size());

    size_t i = 0;
    while (i < input.size()) {
        unsigned char ch = static_cast<unsigned char>(input[i]);

        // Handle UTF-8 multi-byte sequences
        if ((ch & 0x80) == 0) {
            // ASCII character (0-127), pass through except space mapping
            result.push_back(ch);
            i++;
        }
        else if ((ch & 0xE0) == 0xC0) {
            // 2-byte UTF-8 sequence
            if (i + 1 >= input.size()) {
                result.push_back(ch);
                i++;
                continue;
            }

            uint32_t codepoint = ((ch & 0x1F) << 6) | (input[i + 1] & 0x3F);

            auto it = unicode_to_byte.find(codepoint);
            if (it != unicode_to_byte.end()) {
                // Found a mapping, use original byte
                result.push_back(static_cast<char>(it->second));
            } else {
                // No mapping, keep original characters
                result.push_back(input[i]);
                result.push_back(input[i + 1]);
            }
            i += 2;
        }
        else if ((ch & 0xF0) == 0xE0) {
            // 3-byte UTF-8 sequence
            if (i + 2 >= input.size()) {
                result.push_back(ch);
                i++;
                continue;
            }

            uint32_t codepoint = ((ch & 0x0F) << 12) |
                                ((input[i + 1] & 0x3F) << 6) |
                                (input[i + 2] & 0x3F);

            auto it = unicode_to_byte.find(codepoint);
            if (it != unicode_to_byte.end()) {
                result.push_back(static_cast<char>(it->second));
            } else {
                // No mapping, keep original characters
                result.push_back(input[i]);
                result.push_back(input[i + 1]);
                result.push_back(input[i + 2]);
            }
            i += 3;
        }
        else if ((ch & 0xF8) == 0xF0) {
            // 4-byte UTF-8 sequence (not used in GPT-2 mappings)
            if (i + 3 >= input.size()) {
                result.push_back(ch);
                i++;
                continue;
            }

            // Just pass through
            result.push_back(input[i]);
            result.push_back(input[i + 1]);
            result.push_back(input[i + 2]);
            result.push_back(input[i + 3]);
            i += 4;
        }
        else {
            // Invalid UTF-8 or continuation byte, just pass through
            result.push_back(ch);
            i++;
        }
    }

    return result;
}

void readVocab(const char *fp, unordered_map<string, uint32_t> &vocab) {
  ifstream file(fp);
  if (!file) {
    fprintf(stderr, "Cannot open vocabulary file %s\n", fp);
    exit(EXIT_FAILURE);
  }
  fprintf(stderr, "Loading vocabulary from %s ...\n", fp);
  string line, key;
  uint64_t total = 0;
  while (getline(file, line)) {
    vector<string> splits;
    split(splits, line, ' ');
    assert(splits.size() == 2);
    key = decode_gpt2_display_chars(splits[0]);
    assert(vocab.find(key) == vocab.end());
    int count = stoi(splits[1]);
    vocab[key] = count;
    total += count;
  }
  fprintf(stderr, "Read %lu words (%lu unique) from vocabulary file.\n", total,
          vocab.size());
}

void readCodes(const char *fp, unordered_map<tps, uint32_t, pair_hash> &codes,
               unordered_map<string, tps> &reversed_codes) {
  ifstream file(fp);
  if (!file) {
    fprintf(stderr, "Cannot open codes file %s\n", fp);
    exit(EXIT_FAILURE);
  }
  fprintf(stderr, "Loading codes from %s ...\n", fp);
  string line;
  while (getline(file, line)) {
    vector<string> splits;
    string pair0, pair1;

    split(splits, line, ' ');
    assert(splits.size() == 3);
    pair0 = decode_gpt2_display_chars(splits[0]);
    pair1 = decode_gpt2_display_chars(splits[1]);
    auto pair = make_pair(pair0, pair1);
    string concat = pair0 + pair1;
    assert(codes.find(pair) == codes.end());
    assert(reversed_codes.find(concat) == reversed_codes.end());
    codes[pair] = codes.size();
    reversed_codes[concat] = pair;
  }
  fprintf(stderr, "Read %lu codes from the codes file.\n", codes.size());
}

void decompose(const string s, vector<string> &newSubwords,
               const unordered_map<string, tps> &reversed_codes,
               const unordered_map<string, uint32_t> &vocab, bool isFinal) {
  auto it = reversed_codes.find(s);
  if (it == reversed_codes.end()) {
    // TODO this whole block below is just some sanity check
    // if we cannot un-merge a subword, it has to be a char
    string s2 = isFinal ? s.substr(0, s.size() - kEndWordLength) : s;
    int count = 0;
    for (size_t j = 0; j < s2.size(); j++) {
      if ((s2[j] & 0xc0) != 0x80) {
        count++;
      }
    }
    assert(count == 1);
    newSubwords.push_back(s);
    return;
  }
  assert(it != reversed_codes.end());
  string token1 = it->second.first;
  if (vocab.find(token1 + kTokenDelim) == vocab.end()) {
    decompose(token1, newSubwords, reversed_codes, vocab, false);
  } else {
    newSubwords.push_back(token1);
  }
  string token2 = it->second.second;
  auto query = token2 + kTokenDelim;
  if (isFinal) {
    query = token2.substr(0, token2.size() - kEndWordLength);
  }
  if (vocab.find(query) == vocab.end()) {
    decompose(token2, newSubwords, reversed_codes, vocab, isFinal);
  } else {
    newSubwords.push_back(token2);
  }
}

void limitVocab(const vector<string> &subwords, vector<string> &newSubwords,
                const unordered_map<string, tps> &reversed_codes,
                const unordered_map<string, uint32_t> &vocab) {
  string query;
  for (size_t i = 0; i < subwords.size(); i++) {
    bool isFinal = i == subwords.size() - 1;
    auto &subword = subwords[i];
    if (isFinal) {
      query = subword.substr(0, subword.size() - kEndWordLength);
    } else {
      query = subword + kTokenDelim;
    }
    if (vocab.find(query) == vocab.end()) {
      decompose(subword, newSubwords, reversed_codes, vocab, isFinal);
    } else {
      newSubwords.push_back(subword);
    }
  }
}

string process_bpe(vector<string> &subwords,
                   unordered_map<tps, uint32_t, pair_hash> &codes,
                   unordered_map<string, tps> &reversed_codes,
                   unordered_map<string, uint32_t> &vocab) {
  // merge subWords as much as possible
  vector<string> newSubwords;
  while (subwords.size() > 1) {
    // find the best pair
    int bestPairId = -1;
    auto bestPair = codes.end(); // TODO ugly hack that works
    for (size_t i = 0; i < subwords.size() - 1; i++) {
      auto pair = make_pair(subwords[i], subwords[i + 1]);
      auto it = codes.find(pair);
      int pairRank = it == codes.end() ? -1 : it->second;
      if (pairRank >= 0 && (bestPairId == -1 || int(bestPair->second) > pairRank)) {
        bestPair = it;
        bestPairId = i;
      }
    }
    // if we cannot merge anything, stop
    if (bestPairId == -1) {
      break;
    }
    // otherwise, merge subWords
    bool justMerged = false;
    newSubwords = vector<string>();
    for (size_t i = 0; i < subwords.size(); i++) {
      if ((i + 1 < subwords.size()) && (not justMerged) &&
          subwords[i] == bestPair->first.first &&
          subwords[i + 1] == bestPair->first.second) {
        newSubwords.push_back(subwords[i] + subwords[i + 1]);
        justMerged = true;
      } else {
        if (not justMerged) {
          newSubwords.push_back(subwords[i]);
        }
        justMerged = false;
      }
    }
    subwords = newSubwords;
  }
  // check that we are only using words in the dictionary
  if (vocab.size() > 0) {
    vector<string> newSubwords;
    limitVocab(subwords, newSubwords, reversed_codes, vocab);
    subwords = newSubwords;
  }
  // concat subWords
  string result;
  for (auto x : subwords) {
    result = result + x + kTokenDelim + " ";
  }
  return result.substr(
    0,
    result.size() - kEndWordLength - kTokenDelimLength - 1 // "</w>@@ "
  );
}

void applybpe(const char *outputFile, const char *inputFile,
              const char *codesPath, const char *vocabPath) {
  using namespace std;

  auto start = chrono::steady_clock::now();

  auto t_after_vocab  = start;
  auto t_after_codes  = start;
  auto t_after_read   = start;
  auto t_after_token  = start;
  auto t_after_bpe    = start;
  auto t_after_output = start;

  // ===== READ vocab =====
  unsigned long sz = 0;
  std::unordered_map<std::string, uint32_t> vocab;
  if (strcmp(vocabPath, "") != 0) {
    readVocab(vocabPath, vocab);
  }
  t_after_vocab = std::chrono::steady_clock::now();

  // ===== READ codes =====
  std::unordered_map<tps, uint32_t, pair_hash> codes;
  std::unordered_map<std::string, tps> reversed_codes;
  readCodes(codesPath, codes, reversed_codes);
  t_after_codes = std::chrono::steady_clock::now();

  // ===== MPI rank / size =====
  int mpi_rank = 0;
  int mpi_size = 1;
#ifdef CONFIG_MPI
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &mpi_rank);
  MPI_Comm_size(comm, &mpi_size);
#endif

// ===== READ input=====
std::unordered_map<std::string, uint32_t> word_count;

#ifdef CONFIG_MPI
  sz = fastBPE::readText_mpi_parallel(inputFile, word_count, comm);

  std::vector<char> wc_buffer;
  if (mpi_rank == 0) {
    serialize_word_count(word_count, wc_buffer);
  }

  std::uint64_t wc_size = (mpi_rank == 0)
                            ? static_cast<std::uint64_t>(wc_buffer.size())
                            : 0;
  MPI_Bcast(&wc_size, 1, MPI_UINT64_T, 0, comm);

  if (mpi_rank != 0) {
    wc_buffer.resize(static_cast<std::size_t>(wc_size));
  }
  if (wc_size > 0) {
    MPI_Bcast(wc_buffer.data(), static_cast<int>(wc_size), MPI_BYTE, 0, comm);
  }

  if (mpi_rank != 0) {
    deserialize_word_count(wc_buffer.data(), wc_buffer.size(), word_count);
  }

#else
  sz = readText(inputFile, word_count);
#endif


  t_after_read = std::chrono::steady_clock::now();

  // ===== tokenize=====
  std::unordered_map<std::string, std::vector<std::string>> bpeTok;
  tokenize_str(word_count, bpeTok);
  t_after_token = std::chrono::steady_clock::now();

  vector<pair<string, vector<string>>> bpeTokVec;
  bpeTokVec.reserve(bpeTok.size());
  for (auto &x : bpeTok) {
    bpeTokVec.push_back(x);
  }

  // ===== MPI rank / size &  chunk =====


#ifdef CONFIG_MPI
  MPI_Comm_rank(comm, &mpi_rank);
  MPI_Comm_size(comm, &mpi_size);
#endif

  size_t begin = 0;
  size_t end   = bpeTokVec.size();

#ifdef CONFIG_MPI
  {
    size_t total = bpeTokVec.size();
    size_t base  = (mpi_size > 0) ? (total / mpi_size) : total;
    size_t rem   = (mpi_size > 0) ? (total % mpi_size) : 0;

    if (mpi_rank < rem) {
      begin = mpi_rank * (base + 1);
      end   = begin + (base + 1);
    } else {
      begin = mpi_rank * base + rem;
      end   = begin + base;
    }
  }
#endif

  unordered_map<string, string> local_bpe;

  // ====== BPE compute ======
#ifndef CONFIG_OMP
  cout << "Spawning " << kThreads << " threads on MPI rank "
       << mpi_rank << "/" << mpi_size << endl;

  unordered_map<string, string> bpe[kThreads];
  vector<thread> threads;
  threads.reserve(kThreads);

  for (size_t i = 0; i < kThreads; i++) {
    threads.emplace_back(
      [&](size_t this_thread) {
        for (size_t w = begin + this_thread; w < end; w += kThreads) {
          auto &x = bpeTokVec[w];
          bpe[this_thread][x.first] =
              process_bpe(x.second, codes, reversed_codes, vocab);
        }
      },
      i
    );
  }

  for (size_t i = 0; i < kThreads; i++) {
    threads[i].join();
    for (auto &kv : bpe[i]) {
      local_bpe.emplace(std::move(kv.first), std::move(kv.second));
    }
  }

#else // CONFIG_OMP

  // --- OMP variants ---
#if defined(CONFIG_OMP_CRITICAL)
  int nr_threads = 1;
  #pragma omp parallel
  {
    if (omp_get_thread_num() == 0) {
      nr_threads = omp_get_num_threads();
    }
  }

  cout << "omp critical region, MPI rank "
       << mpi_rank << "/" << mpi_size
       << ", number of threads = " << nr_threads << endl;

  #pragma omp parallel for
  for (size_t w = begin; w < end; w++) {
    auto &x = bpeTokVec[w];
    auto str = process_bpe(x.second, codes, reversed_codes, vocab);
    #pragma omp critical
    {
      local_bpe[x.first] = std::move(str);
    }
  }

#elif defined(CONFIG_OMP_SINGLE_THREADED_MERGE)

  int nr_threads = 1;
  #pragma omp parallel
  {
    if (omp_get_thread_num() == 0) {
      nr_threads = omp_get_num_threads();
    }
  }

  cout << "omp single thread merge, MPI rank "
       << mpi_rank << "/" << mpi_size
       << ", number of threads = " << nr_threads << endl;

  vector<unordered_map<string, string>> bpe(nr_threads);
  const size_t local_size      = (end > begin) ? (end - begin) : 0;
  const size_t approx_per_thr  = (nr_threads > 0) ? (local_size / nr_threads) : local_size;

  for (int i = 0; i < nr_threads; ++i) {
    bpe[i].reserve(approx_per_thr + 1);
  }

  #pragma omp parallel for
  for (size_t w = begin; w < end; w++) {
    auto &x = bpeTokVec[w];
    auto str = process_bpe(x.second, codes, reversed_codes, vocab);
    bpe[omp_get_thread_num()][x.first] = std::move(str);
  }

  for (int i = 0; i < nr_threads; i++) {
    for (auto &kv : bpe[i]) {
      local_bpe.emplace(std::move(kv.first), std::move(kv.second));
    }
  }

#elif defined(CONFIG_OMP_TBB)

  int nr_threads = 1;
  #pragma omp parallel
  {
    if (omp_get_thread_num() == 0) {
      nr_threads = omp_get_num_threads();
    }
  }

  cout << "omp+tbb, MPI rank "
       << mpi_rank << "/" << mpi_size
       << ", number of threads = " << nr_threads << endl;

  tbb::concurrent_unordered_map<string, string> concurrent_bpe(
      (end > begin) ? (end - begin) : 0);

  #pragma omp parallel for
  for (size_t w = begin; w < end; w++) {
    auto &x = bpeTokVec[w];
    auto str = process_bpe(x.second, codes, reversed_codes, vocab);
    concurrent_bpe[x.first] = std::move(str);
  }

  for (auto &kv : concurrent_bpe) {
    local_bpe.emplace(kv.first, std::move(kv.second));
  }

#else
#error "Define a parallel method (CONFIG_OMP_CRITICAL / _SINGLE_THREADED_MERGE / _TBB)"
#endif // OMP variants
#endif // CONFIG_OMP

  t_after_bpe = chrono::steady_clock::now();

  // ===== MPI gather + output =====
#ifdef CONFIG_MPI


  std::vector<char> sendbuf;
  serialize_map(local_bpe, sendbuf);
  int send_count = static_cast<int>(sendbuf.size());


  std::vector<int> recv_counts;
  std::vector<int> displs;
  std::vector<char> recvbuf;

  if (mpi_rank == 0) {
    recv_counts.resize(mpi_size);
  }

  MPI_Gather(&send_count, 1, MPI_INT,
             mpi_rank == 0 ? recv_counts.data() : nullptr, 1, MPI_INT,
             0, comm);

  int total_bytes = 0;
  if (mpi_rank == 0) {
    displs.resize(mpi_size);
    displs[0] = 0;
    for (int i = 0; i < mpi_size; ++i) {
      total_bytes += recv_counts[i];
      if (i > 0) {
        displs[i] = displs[i - 1] + recv_counts[i - 1];
      }
    }
    recvbuf.resize(total_bytes);
  }


  MPI_Gatherv(sendbuf.data(), send_count, MPI_BYTE,
              mpi_rank == 0 ? recvbuf.data()   : nullptr,
              mpi_rank == 0 ? recv_counts.data(): nullptr,
              mpi_rank == 0 ? displs.data()    : nullptr,
              MPI_BYTE,
              0, comm);

  if (mpi_rank == 0) {
    unordered_map<string, string> final_bpe;
    final_bpe.reserve(bpeTokVec.size());


    for (int r = 0; r < mpi_size; ++r) {
      int offset = displs[r];
      int len    = recv_counts[r];
      if (len <= 0) continue;
      deserialize_map_segment(
        recvbuf.data() + offset,
        static_cast<std::size_t>(len),
        final_bpe);
    }

    outputText(outputFile, inputFile, final_bpe);
  }

  t_after_output = chrono::steady_clock::now();

#else  // No MPI：local_bpe is final_bpe

  {
    unordered_map<string, string> &final_bpe = local_bpe;
    outputText(outputFile, inputFile, final_bpe);
  }
  t_after_output = chrono::steady_clock::now();

#endif // CONFIG_MPI

  // ===== time and bandwith=====
  using us = chrono::microseconds;

  double dt_vocab_local  = chrono::duration_cast<us>(t_after_vocab  - start).count();
  double dt_codes_local  = chrono::duration_cast<us>(t_after_codes  - t_after_vocab).count();
  double dt_read_local   = chrono::duration_cast<us>(t_after_read   - t_after_codes).count();
  double dt_token_local  = chrono::duration_cast<us>(t_after_token  - t_after_read).count();
  double dt_bpe_local    = chrono::duration_cast<us>(t_after_bpe    - t_after_token).count();
  double dt_output_local = chrono::duration_cast<us>(t_after_output - t_after_bpe).count();
  double dt_total_local  = chrono::duration_cast<us>(t_after_output - start).count();

  auto safe_bw = [](unsigned long bytes, double us) {
    if (us <= 0.0) return 0.0;
    return (double)bytes / us * 1e6; // B/s
  };

#ifdef CONFIG_MPI
  double dt_vocab_max  = 0, dt_codes_max  = 0, dt_read_max   = 0;
  double dt_token_max  = 0, dt_bpe_max    = 0, dt_output_max = 0;
  double dt_total_max  = 0;

  MPI_Reduce(&dt_vocab_local,  &dt_vocab_max,  1, MPI_DOUBLE, MPI_MAX, 0, comm);
  MPI_Reduce(&dt_codes_local,  &dt_codes_max,  1, MPI_DOUBLE, MPI_MAX, 0, comm);
  MPI_Reduce(&dt_read_local,   &dt_read_max,   1, MPI_DOUBLE, MPI_MAX, 0, comm);
  MPI_Reduce(&dt_token_local,  &dt_token_max,  1, MPI_DOUBLE, MPI_MAX, 0, comm);
  MPI_Reduce(&dt_bpe_local,    &dt_bpe_max,    1, MPI_DOUBLE, MPI_MAX, 0, comm);
  MPI_Reduce(&dt_output_local, &dt_output_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
  MPI_Reduce(&dt_total_local,  &dt_total_max,  1, MPI_DOUBLE, MPI_MAX, 0, comm);

  if (mpi_rank == 0) {
    double bw_read   = safe_bw(sz, dt_read_max);
    double bw_bpe    = safe_bw(sz, dt_bpe_max);
    double bw_output = safe_bw(sz, dt_output_max);
    double bw_total  = safe_bw(sz, dt_total_max);

    cout << "===== Timing (MPI, max across ranks, Nprocs="
         << mpi_size << ") =====" << endl;
    cout << "readVocab:      " << dt_vocab_max  / 1e3 << " ms" << endl;
    cout << "readCodes:      " << dt_codes_max  / 1e3 << " ms" << endl;
    cout << "readText:       " << dt_read_max   / 1e3 << " ms, BW ~= "
         << bw_read  / (1024.0*1024.0) << " MB/s" << endl;
    cout << "tokenize:       " << dt_token_max  / 1e3 << " ms" << endl;
    cout << "BPE compute:    " << dt_bpe_max    / 1e3 << " ms, BW ~= "
         << bw_bpe   / (1024.0*1024.0) << " MB/s" << endl;
    cout << "gather+output:  " << dt_output_max / 1e3 << " ms, BW ~= "
         << bw_output / (1024.0*1024.0) << " MB/s" << endl;
    cout << "-----------------------------------------" << endl;
    cout << "Total:          " << dt_total_max  / 1e3 << " ms, BW ~= "
         << bw_total / (1024.0*1024.0) << " MB/s" << endl;
  }

#else
  double bw_read   = safe_bw(sz, dt_read_local);
  double bw_bpe    = safe_bw(sz, dt_bpe_local);
  double bw_output = safe_bw(sz, dt_output_local);
  double bw_total  = safe_bw(sz, dt_total_local);

  cout << "===== Timing (single process) =====" << endl;
  cout << "readVocab:      " << dt_vocab_local  / 1e3 << " ms" << endl;
  cout << "readCodes:      " << dt_codes_local  / 1e3 << " ms" << endl;
  cout << "readText:       " << dt_read_local   / 1e3 << " ms, BW ~= "
       << bw_read  / (1024.0*1024.0) << " MB/s" << endl;
  cout << "tokenize:       " << dt_token_local  / 1e3 << " ms" << endl;
  cout << "BPE compute:    " << dt_bpe_local    / 1e3 << " ms, BW ~= "
       << bw_bpe   / (1024.0*1024.0) << " MB/s" << endl;
  cout << "outputText:     " << dt_output_local / 1e3 << " ms, BW ~= "
       << bw_output / (1024.0*1024.0) << " MB/s" << endl;
  cout << "-----------------------------------" << endl;
  cout << "Total:          " << dt_total_local  / 1e3 << " ms, BW ~= "
       << bw_total / (1024.0*1024.0) << " MB/s" << endl;
#endif
}



class BPEApplyer {
private:
  _hash_map<string, uint32_t> vocab;
  _hash_map<tps, uint32_t, pair_hash> codes;
  _hash_map<string, tps> reversed_codes;

public:
  BPEApplyer(const string& codesPath, const string& vocabPath) {
    if (vocabPath.size() > 0) readVocab(vocabPath.c_str(), vocab);
    readCodes(codesPath.c_str(), codes, reversed_codes);
  }

  vector<string> apply(vector<string>& sentences) {
    vector<string> res;
    for(auto &s: sentences) {
      res.emplace_back("");
      string& cur = res.back();
      vector<string> words;
      split(words, s, ' ');
      for (size_t i = 0; i < words.size(); i++) {
        auto word = words[i];
        vector<string> word_bpes;
        int pos = 0, realLength = 0;
        int lastStart = 0;
        while (word[pos]) {
          bool newChar = (word[pos] & 0xc0) != 0x80; // not a continuation byte
          realLength += newChar;
          if (newChar && pos > 0) {
            auto new_token = word.substr(lastStart, pos - lastStart);
            word_bpes.push_back(new_token);
            lastStart = pos;
          }
          pos++;
        }
        auto bpe = word.substr(lastStart, string::npos) + kEndWord;
        word_bpes.push_back(bpe);
        cur += process_bpe(word_bpes, codes, reversed_codes, vocab);
        if (i < words.size() - 1) cur += " ";
      }
    }
    return res;
  }

};


void applybpe_stream(const char *codesPath, const char *vocabPath) {
  BPEApplyer applyer(codesPath, vocabPath);
  std::string line;
  while(std::getline(std::cin, line)) {
    vector<string> tmp;
    tmp.push_back(line);
    for(auto& l : applyer.apply(tmp)){
      std::cout << l << std::endl;
    }
  }
}

};
