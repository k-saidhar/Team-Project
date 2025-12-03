#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <lzma.h>
#include <cstring>
#include <xxhash.h>
#include <mutex>
#include <shared_mutex>
#include <sstream>

std::shared_mutex cache_mutex;
std::unordered_map<std::string, uint64_t> cx_cache;

uint64_t get_cx(const std::string& key) {
    std::shared_lock lock(cache_mutex);
    auto it = cx_cache.find(key);
    return it != cx_cache.end() ? it->second : 0;
}

void set_cx(const std::string& key, uint64_t size) {
    std::unique_lock lock(cache_mutex);
    cx_cache[key] = size;
}

uint64_t lzma_compress_size(const uint8_t* data, size_t len) {
    lzma_stream strm = LZMA_STREAM_INIT;
    if (lzma_easy_encoder(&strm, 6, LZMA_CHECK_CRC64) != LZMA_OK)
        return len;

    std::vector<uint8_t> out(2 * len + 1024);
    strm.next_in = data;
    strm.avail_in = len;
    strm.next_out = out.data();
    strm.avail_out = out.size();

    lzma_code(&strm, LZMA_FINISH);
    uint64_t compressed = strm.total_out;
    lzma_end(&strm);
    return compressed;
}

constexpr int NUM_HASH = 64;
uint64_t minhash(const uint8_t* data, size_t len) {
    uint64_t sig[NUM_HASH];
    std::fill(sig, sig + NUM_HASH, ULLONG_MAX);

    for (size_t i = 0; i + 8 <= len; ++i) {
        uint64_t h = XXH64(data + i, 8, 0x12345678);
        for (int j = 0; j < NUM_HASH; ++j) {
            uint64_t ph = h ^ (j * 0x9e3779b97f4a7c15ULL);
            if (ph < sig[j]) sig[j] = ph;
        }
    }
    uint64_t final = 0;
    for (int j = 0; j < NUM_HASH; ++j) final ^= sig[j];
    return final;
}

std::pair<const uint8_t*, size_t> mmap_file(const std::string& path) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd == -1) return {nullptr, 0};
    struct stat sb;
    fstat(fd, &sb);
    void* ptr = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (ptr == MAP_FAILED) return {nullptr, 0};
    return {(const uint8_t*)ptr, sb.st_size};
}

int main(int argc, char** argv) {
    if (argc < 6) return 1;
    std::string mode = argv[1];
    std::string folder = argv[2];

    if (mode == "assign") {
        std::string file_list = argv[3];
        std::string proto_list = argv[4];
        double threshold = std::stod(argv[5]);

        std::vector<std::string> files, proto_paths;
        std::vector<int> proto_ids;

        std::istringstream fl(file_list);
        std::string f;
        while (std::getline(fl, f, ',')) if (!f.empty()) files.push_back(f);

        std::istringstream pl(proto_list);
        std::string token;
        while (std::getline(pl, token, ',')) {
            size_t sep = token.find('|');
            if (sep == std::string::npos) continue;
            proto_paths.push_back(token.substr(0, sep));
            proto_ids.push_back(std::stoi(token.substr(sep + 1)));
        }

        std::vector<std::pair<const uint8_t*, size_t>> proto_mmaps;
        std::vector<uint64_t> proto_minhash;

        for (const auto& p : proto_paths) {
            auto [data, len] = mmap_file(folder + "/" + p);
            if (!data) continue;
            proto_mmaps.emplace_back(data, len);
            proto_minhash.push_back(minhash(data, len));
        }

        for (const auto& fname : files) {
            auto [data, len] = mmap_file(folder + "/" + fname);
            if (!data) {
                std::cout << fname << "|-1|999.0\n";
                continue;
            }

            uint64_t file_mh = minhash(data, len);
            uint64_t Cx = get_cx(fname);
            if (Cx == 0) {
                Cx = lzma_compress_size(data, len);
                set_cx(fname, Cx);
            }

            double best_dist = 999.0;
            int best_id = -1;

            for (size_t i = 0; i < proto_paths.size(); ++i) {
                double mh_dist = __builtin_popcountll(file_mh ^ proto_minhash[i]) / 64.0;
                if (mh_dist > 0.6) continue;

                uint64_t Cy = get_cx(proto_paths[i]);
                if (Cy == 0) {
                    Cy = lzma_compress_size(proto_mmaps[i].first, proto_mmaps[i].second);
                    set_cx(proto_paths[i], Cy);
                }

                size_t xy_len = len + proto_mmaps[i].second;
                std::vector<uint8_t> xy(xy_len);
                memcpy(xy.data(), data, len);
                memcpy(xy.data() + len, proto_mmaps[i].first, proto_mmaps[i].second);
                uint64_t Cxy = lzma_compress_size(xy.data(), xy_len);

                double ncd = (Cxy - std::min(Cx, Cy)) / double(std::max(Cx, Cy));
                if (ncd < best_dist) {
                    best_dist = ncd;
                    best_id = proto_ids[i];
                }
            }

            std::cout << fname << "|" << best_id << "|" << best_dist << "\n";
            munmap((void*)data, len);
        }

        for (auto& p : proto_mmaps)
            if (p.first) munmap((void*)p.first, p.second);
    }
    return 0;
}