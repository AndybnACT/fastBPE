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
#include <mpi.h>
#include <boost/mpi.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/unordered_map.hpp>
namespace mpi = boost::mpi;
#endif


namespace fastBPE {

using namespace std;

const size_t kMaxPairs = 1000 * 1000 * 1000;
size_t kThreads = max(1, min(10, int(thread::hardware_concurrency())));
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

size_t readText_split(const char *fp, vector<_hash_map<string, uint32_t>> &all_word_count) {
  uint64_t total = 0;
  size_t sz = 0;

  if (string(fp).compare("-") == 0) {
    fprintf(stderr, "Not implemented\n");
    exit(EXIT_FAILURE);
  }
  else {
    int fd = safeOpen(fp, O_RDONLY);

    struct stat s;
    fstat(fd, &s);
    fprintf(stderr, "Loading vocabulary from %s ...\n", fp);

    size_t size = s.st_size;
    char *f = (char *)mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    sz = size;

#ifdef CONFIG_MPI
    mpi::communicator world;

    vector<size_t> rank_boundary = get_boundary(f, size, world.size());
    size = rank_boundary[world.rank() + 1] - rank_boundary[world.rank()];
    f += rank_boundary[world.rank()];
#endif /* CONFIG_MPI  */


    vector<size_t> boundary = get_boundary(f, size, kThreads);
//    for (int i = 0; i < kThreads; i++) {
//      cout << "start of " << i << "th thread " << boundary[i] << endl;
//    }
    #pragma omp parallel reduction(+: total)
    {
      size_t id = omp_get_thread_num();
      string cur_word;
      _hash_map<string, uint32_t> &word_count = all_word_count[id];

      for (size_t i = boundary[id]; i < boundary[id + 1]; i++) {
        if (f[i] == ' ' || f[i] == '\n') {
          if (cur_word.size() == 0)
            continue;
          // end of word
          auto it = word_count.find(cur_word);
          int count = it != word_count.end() ? it->second : 0;
          word_count[cur_word] = count + 1;
          total++;
          cur_word.clear();
        } else {
          cur_word.push_back(f[i]);
        }
      }
    }
  }

  fprintf(stderr, "Read %lu words from text file.\n", total);
  return sz;
}

size_t readText(const char *fp, _hash_map<string, uint32_t> &word_count) {
  string cur;
  uint64_t total = 0;
  size_t sz = 0;
  auto deal_with_char = [&](char cur_char, string &cur_word){
    if (cur_char == ' ' || cur_char == '\n') {
      if (cur_word.size() == 0)
        return;
      // end of word
      auto it = word_count.find(cur_word);
      int count = it != word_count.end() ? it->second : 0;
      word_count[cur_word] = count + 1;
      total++;
      cur_word.clear();
    } else {
      cur_word.push_back(cur_char);
    }
  };

  if (string(fp).compare("-") == 0) {
    for (std::string line; std::getline(std::cin, line);) {
      for(char c: line){
        deal_with_char(c, cur);
      }
      deal_with_char('\n', cur);
    }
  }
  else {
    int fd = safeOpen(fp, O_RDONLY);

    struct stat s;
    fstat(fd, &s);
    fprintf(stderr, "Loading vocabulary from %s ...\n", fp);

    size_t size = s.st_size;
    char *f = (char *)mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);

    for (size_t i = 0; i < size; i++) {
      deal_with_char(f[i], cur);
    }

    sz = size;
  }
  fprintf(stderr, "Read %lu words (%lu unique) from text file.\n", total,
          word_count.size());
  return sz;
}

std::pair<size_t, uint64_t> output_or_count(
  vector<_hash_map<string, string>> &bpe, size_t size, char *f, vector<char> *allbuf
) {
  uint64_t sz = 0, total = 0;
  vector<size_t> boundary = get_boundary(f, size, kThreads);

  #pragma omp parallel reduction(+:sz, total)
  {
    string cur_word;
    size_t id = omp_get_thread_num();
    vector<char> *buf = &allbuf[id];

    for (size_t i = boundary[id]; i < boundary[id + 1]; i++) {
      auto &cur_char = f[i];
      if (cur_char == ' ' || cur_char == '\n') {
        if (cur_word.size() == 0) {
          buf->push_back(cur_char);
	  sz++;
          continue;
        }
        // end of word : write bpe to output
        auto it = bpe[id].find(cur_word);
        assert(it != bpe[id].end());
        for (auto x : it->second) {
          buf->push_back(x);
	  sz++;
        }
        buf->push_back(cur_char);
	sz++;

        total++;
        cur_word.clear();
      } else {
        cur_word.push_back(cur_char);
      }
    }
  }

  return std::make_pair(sz, total);
}

void outputText(const char *fpo, const char *fp,
                vector<_hash_map<string, string>> &bpe) {

  int fd = safeOpen(fp, O_RDONLY);
  auto fdOut = safeOpen(fpo, O_RDWR | O_CREAT | O_TRUNC, 0666);

  struct stat s;
  fstat(fd, &s);

  fprintf(stderr, "Applying BPE to %s (Parallelized) ...\n", fp);
  auto size = s.st_size;
  char *f = (char *)mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
  vector<char> allbuf[kThreads];
  vector<size_t> off(kThreads);

#ifdef CONFIG_MPI
    mpi::communicator world;

    vector<size_t> rank_boundary = get_boundary(f, size, world.size());
    size = rank_boundary[world.rank() + 1] - rank_boundary[world.rank()];
    f += rank_boundary[world.rank()];
#endif /* CONFIG_MPI  */

  auto p = output_or_count(bpe, size, f, allbuf);
  size_t out_size = p.first;

  if (ftruncate(fdOut, out_size) < 0) {
    fprintf(stderr, "Couldn't truncate output file %s to size %lu\n", fpo,
            out_size);
    exit(EXIT_FAILURE);
  }

  off[0] = 0;
  for (int i = 1; i < kThreads; i++) {
    off[i] = allbuf[i - 1].size() + off[i - 1];
  }


  char *fo = (char *)mmap(NULL, out_size, PROT_WRITE, MAP_SHARED, fdOut, 0);
  if (fo == MAP_FAILED) {
    fprintf(stderr, "Output memory map failed : %d.\n", errno);
    exit(EXIT_FAILURE);
  }

#pragma omp parallel for
  for (int i = 0; i < kThreads; i++) {
    memcpy(fo+off[i], allbuf[i].data(), allbuf[i].size());
  }

  fprintf(stderr, "Modified %lu words from text file.\n", p.second);
  munmap(fo, out_size);
  munmap(f, s.st_size);
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

void tokenize(const _hash_map<string, uint32_t> &word_count,
              _hash_map<string, uint32_t> &token_to_int,
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

void tokenize_str(const _hash_map<string, uint32_t> &word_count,
                  _hash_map<string, vector<string>> &words) {
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
using pc = _hash_map<tp, pair<int32_t, tp> *, pair_hash>;

void count_in_word(
    list<uint32_t> &word, uint32_t wi, uint32_t count, pc &pair_counts,
    vector<pair<int32_t, tp>> &contiguous_counts,
    _hash_map<tp, unordered_set<uint32_t>, pair_hash> &where) {
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
  _hash_map<string, uint32_t> word_count;
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
  _hash_map<string, uint32_t> word_count;
  readText(inputFile1, word_count);
  if (strcmp(inputFile2, "") != 0) {
    readText(inputFile2, word_count);
  }

  // a token is an int, it represents a string
  _hash_map<string, uint32_t> token_to_int;
  vector<string> int_to_token;

  vector<list<uint32_t>> words;
  vector<int32_t> counts;

  tokenize(word_count, token_to_int, int_to_token, words, counts);

  vector<pair<int32_t, tp>> contiguous_counts;
  contiguous_counts.reserve(kMaxPairs);

  pc pair_counts;
  _hash_map<tp, unordered_set<uint32_t>, pair_hash> where_to_update;

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
    static const _hash_map<uint32_t, uint8_t> unicode_to_byte = {
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

void readVocab(const char *fp, _hash_map<string, uint32_t> &vocab) {
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

void readCodes(const char *fp, _hash_map<tps, uint32_t, pair_hash> &codes,
               _hash_map<string, tps> &reversed_codes) {
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
               const _hash_map<string, tps> &reversed_codes,
               const _hash_map<string, uint32_t> &vocab, bool isFinal) {
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
                const _hash_map<string, tps> &reversed_codes,
                const _hash_map<string, uint32_t> &vocab) {
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
                   _hash_map<tps, uint32_t, pair_hash> &codes,
                   _hash_map<string, tps> &reversed_codes,
                   _hash_map<string, uint32_t> &vocab) {
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
  // read vocabulary (to which we want to limit the output file)
  auto start = chrono::steady_clock::now();
  unsigned long sz;
  int nr_threads;
  #pragma omp parallel
  {
    if (omp_get_thread_num() == 0) {
      nr_threads = omp_get_num_threads();
    }
  }
  kThreads = nr_threads;

  _hash_map<string, uint32_t> vocab;
  if (strcmp(vocabPath, "") != 0) {
    readVocab(vocabPath, vocab);
  }

  // read codes
  _hash_map<tps, uint32_t, pair_hash> codes;
  _hash_map<string, tps> reversed_codes;
  readCodes(codesPath, codes, reversed_codes);

  // read input file words
  vector<_hash_map<string, uint32_t>> word_count(nr_threads);

  sz = readText_split(inputFile, word_count);

  vector<_hash_map<string, string>> final_bpe(nr_threads);
  #pragma omp parallel
  {
    int id = omp_get_thread_num();
    _hash_map<string, string> &bpe = final_bpe[id];
    // tokenize
    _hash_map<string, vector<string>> bpeTok;
    tokenize_str(word_count[id], bpeTok);

    vector<pair<string, vector<string>>> bpeTokVec;
    for (auto x : bpeTok) {
      bpeTokVec.push_back(x);
    }

    for (size_t w = 0; w < bpeTokVec.size(); w++) {
      auto &x = bpeTokVec[w];
      bpe[x.first] = process_bpe(x.second, codes, reversed_codes, vocab);
    }
  }

  // output
#ifdef CONFIG_MPI
  mpi::communicator world;

  outputText((string(outputFile) + "." + to_string(world.rank())).c_str(),
             inputFile, final_bpe);
  world.barrier();
  if (world.rank() == 0) {
    auto end = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<std::chrono::microseconds>(end - start);
    double bw = (double) sz / (double) duration.count() * 1e6f;
    cout << "Ranks = " << world.size() << ", Threads = " << nr_threads;
    cout << ", Time spent = " << duration.count() << " ms, ";
    cout << "Bandwidth = " << bw << " B/sec, " << bw / pow(2,20) << " MB/sec" << endl;
  }
#else /* !CONFIG_MPI  */
  outputText(outputFile, inputFile, final_bpe);
  auto end = chrono::steady_clock::now();
  auto duration = chrono::duration_cast<std::chrono::microseconds>(end - start);
  double bw = (double) sz / (double) duration.count() * 1e6f;
  cout << "Ranks = 1" << ", Threads = " << nr_threads;
  cout << ", Time spent = " << duration.count() << " ms, " ;
  cout << "Bandwidth = " << bw << " B/sec, " << bw / pow(2,20) << " MB/sec" << endl;
#endif /* CONFIG_MPI  */
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
