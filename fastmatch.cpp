// libfastmatch v2: meet-in-the-middle pair matcher with optional pair-PSD screen.
// Build (Linux):  c++ -O3 -std=c++17 -shared -fPIC fastmatch.cpp -o libfastmatch.so
// Build (macOS):  c++ -O3 -std=c++17 -dynamiclib fastmatch.cpp -o libfastmatch.dylib
//
// Finds (i,j,k,l) with wt1*(pafA[i]+pafB[j]) + wt2*(pafC[k]+pafD[l]) == 0.
// Optional screen (enabled when psd pointers non-null and cap > 0):
//   pair (x,y) is skipped iff psdX[x][wX[x]] + psdY[y][wX[x]] > cap.
// Sound: exact matches satisfy the PSD budget at every frequency, so no
// pair belonging to an exact match is ever skipped (see tests_opt.py).
// stats[5] = { pairs_hashed, buckets, probes, hash_collision_checks, psd_skipped }

#include <cstdint>
#include <cstring>
#include <vector>

static inline uint64_t fnv1a(const int16_t *v, int64_t h) {
    uint64_t x = 1469598103934665603ULL;
    const uint8_t *p = (const uint8_t *)v;
    for (int64_t b = 0; b < h * 2; b++) { x ^= p[b]; x *= 1099511628211ULL; }
    return x ? x : 1;
}

extern "C" int64_t match_pairs_v2(
    const int16_t *pafA, int64_t na, const int16_t *pafB, int64_t nb,
    const int16_t *pafC, int64_t nc, const int16_t *pafD, int64_t nd,
    int64_t h, int16_t wt1, int16_t wt2,
    const float *psdA, const int32_t *wA, const float *psdB,
    const float *psdC, const int32_t *wC, const float *psdD,
    int64_t F, float cap1, float cap2,
    int64_t *out_idx, int64_t *stats) {

    for (int i = 0; i < 5; i++) stats[i] = 0;
    if (na * nb == 0 || nc * nd == 0) return 0;
    const bool s1 = psdA && psdB && wA && cap1 > 0;
    const bool s2 = psdC && psdD && wC && cap2 > 0;

    uint64_t cap = 1;
    while ((int64_t)cap < 2 * na * nb) cap <<= 1;
    std::vector<uint64_t> keys(cap, 0);
    std::vector<uint32_t> ai(cap), bj(cap);
    std::vector<int16_t> buf(h);

    for (int64_t i = 0; i < na; i++) {
        const float ax = s1 ? psdA[i * F + wA[i]] : 0.f;
        for (int64_t j = 0; j < nb; j++) {
            if (s1 && ax + psdB[j * F + wA[i]] > cap1 + 1e-6f) {
                stats[4]++; continue;
            }
            for (int64_t s = 0; s < h; s++)
                buf[s] = (int16_t)(wt1 * (pafA[i * h + s] + pafB[j * h + s]));
            uint64_t k = fnv1a(buf.data(), h), pos = k & (cap - 1);
            while (keys[pos] != 0) pos = (pos + 1) & (cap - 1);
            keys[pos] = k; ai[pos] = (uint32_t)i; bj[pos] = (uint32_t)j;
            stats[0]++;
        }
    }
    for (uint64_t p = 0; p < cap; p++) if (keys[p]) stats[1]++;

    std::vector<int16_t> probe(h), chk(h);
    for (int64_t k2 = 0; k2 < nc; k2++) {
        const float cx = s2 ? psdC[k2 * F + wC[k2]] : 0.f;
        for (int64_t l = 0; l < nd; l++) {
            if (s2 && cx + psdD[l * F + wC[k2]] > cap2 + 1e-6f) {
                stats[4]++; continue;
            }
            for (int64_t s = 0; s < h; s++)
                probe[s] = (int16_t)(-wt2 * (pafC[k2 * h + s] + pafD[l * h + s]));
            stats[2]++;
            uint64_t k = fnv1a(probe.data(), h), pos = k & (cap - 1);
            while (keys[pos] != 0) {
                if (keys[pos] == k) {
                    stats[3]++;
                    int64_t i = ai[pos], j = bj[pos];
                    for (int64_t s = 0; s < h; s++)
                        chk[s] = (int16_t)(wt1 * (pafA[i * h + s] + pafB[j * h + s]));
                    if (std::memcmp(chk.data(), probe.data(), h * 2) == 0) {
                        out_idx[0] = i; out_idx[1] = j;
                        out_idx[2] = k2; out_idx[3] = l;
                        return 1;
                    }
                }
                pos = (pos + 1) & (cap - 1);
            }
        }
    }
    return 0;
}

// ===================== v3: monster mode =====================
// Same semantics as match_pairs_v2 for results and for the counters
// pairs_hashed / probes / psd_skipped. Differences (documented):
//   * row combine uses NEON int16 SIMD on ARM64 (scalar C++ elsewhere)
//   * hash is word-at-a-time FNV-1a over u64 chunks (bucketing only;
//     collision-check diagnostics may differ, results cannot -- every
//     hash hit is verified by exact memcmp)
// canon_key: byte-identical reimplementation of core.canonical_key.

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define FM_BACKEND "neon"
#else
#define FM_BACKEND "scalar-c++"
#endif

extern "C" const char *fm_backend() { return FM_BACKEND; }

static inline void combine_rows(const int16_t *a, const int16_t *b,
                                int16_t wt, int16_t *out, int64_t h) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    int64_t s = 0;
    for (; s + 8 <= h; s += 8) {
        int16x8_t va = vld1q_s16(a + s);
        int16x8_t vb = vld1q_s16(b + s);
        vst1q_s16(out + s, vmulq_n_s16(vaddq_s16(va, vb), wt));
    }
    for (; s < h; s++) out[s] = (int16_t)(wt * (a[s] + b[s]));
#else
    for (int64_t s = 0; s < h; s++)
        out[s] = (int16_t)(wt * (a[s] + b[s]));
#endif
}

static inline uint64_t hash64w(const int16_t *v, int64_t h) {
    // FNV-1a over u64 words (tail bytewise). Bucketing only.
    uint64_t x = 1469598103934665603ULL;
    const uint8_t *p = (const uint8_t *)v;
    int64_t nbytes = h * 2, i = 0;
    for (; i + 8 <= nbytes; i += 8) {
        uint64_t w;
        std::memcpy(&w, p + i, 8);
        x ^= w;
        x *= 1099511628211ULL;
    }
    for (; i < nbytes; i++) { x ^= p[i]; x *= 1099511628211ULL; }
    x ^= x >> 29; x *= 0xbf58476d1ce4e5b9ULL; x ^= x >> 32;
    return x ? x : 1;
}

extern "C" int64_t match_pairs_v3(
    const int16_t *pafA, int64_t na, const int16_t *pafB, int64_t nb,
    const int16_t *pafC, int64_t nc, const int16_t *pafD, int64_t nd,
    int64_t h, int16_t wt1, int16_t wt2,
    const float *psdA, const int32_t *wA, const float *psdB,
    const float *psdC, const int32_t *wC, const float *psdD,
    int64_t F, float cap1, float cap2,
    int64_t *out_idx, int64_t *stats) {

    for (int i = 0; i < 5; i++) stats[i] = 0;
    if (na * nb == 0 || nc * nd == 0) return 0;
    const bool s1 = psdA && psdB && wA && cap1 > 0;
    const bool s2 = psdC && psdD && wC && cap2 > 0;

    uint64_t cap = 1;
    while ((int64_t)cap < 2 * na * nb) cap <<= 1;
    std::vector<uint64_t> keys(cap, 0);
    std::vector<uint32_t> ai(cap), bj(cap);
    std::vector<int16_t> buf(h);

    for (int64_t i = 0; i < na; i++) {
        const int16_t *ra = pafA + i * h;
        const float ax = s1 ? psdA[i * F + wA[i]] : 0.f;
        for (int64_t j = 0; j < nb; j++) {
            if (s1 && ax + psdB[j * F + wA[i]] > cap1 + 1e-6f) {
                stats[4]++; continue;
            }
            combine_rows(ra, pafB + j * h, wt1, buf.data(), h);
            uint64_t k = hash64w(buf.data(), h), pos = k & (cap - 1);
            while (keys[pos] != 0) pos = (pos + 1) & (cap - 1);
            keys[pos] = k; ai[pos] = (uint32_t)i; bj[pos] = (uint32_t)j;
            stats[0]++;
        }
    }
    stats[1] = stats[0];  // open addressing: one occupied slot per pair

    std::vector<int16_t> probe(h), chk(h);
    for (int64_t k2 = 0; k2 < nc; k2++) {
        const int16_t *rc = pafC + k2 * h;
        const float cx = s2 ? psdC[k2 * F + wC[k2]] : 0.f;
        for (int64_t l = 0; l < nd; l++) {
            if (s2 && cx + psdD[l * F + wC[k2]] > cap2 + 1e-6f) {
                stats[4]++; continue;
            }
            combine_rows(rc, pafD + l * h, (int16_t)(-wt2),
                         probe.data(), h);
            stats[2]++;
            uint64_t k = hash64w(probe.data(), h), pos = k & (cap - 1);
            while (keys[pos] != 0) {
                if (keys[pos] == k) {
                    stats[3]++;
                    int64_t i = ai[pos], j = bj[pos];
                    combine_rows(pafA + i * h, pafB + j * h, wt1,
                                 chk.data(), h);
                    if (std::memcmp(chk.data(), probe.data(), h * 2) == 0) {
                        out_idx[0] = i; out_idx[1] = j;
                        out_idx[2] = k2; out_idx[3] = l;
                        return 1;
                    }
                }
                pos = (pos + 1) & (cap - 1);
            }
        }
    }
    return 0;
}

// canon_key: byte-identical to core.canonical_key (lexicographic min over
// negation x {rotations} x {reversal} when periodic; negation x reversal
// when not). out must hold n bytes. Returns 0.
extern "C" int canon_key(const int8_t *seq, int64_t n, int periodic,
                         int8_t *out) {
    // Same candidate set as before (negation x reversal x rotations),
    // same lexicographic minimum -- but each rotation is a pointer into a
    // doubled buffer, so a candidate comparison is one memcmp instead of
    // n modulo operations. Byte-identical output, ~70x faster at n=167.
    std::vector<int8_t> bufs(4 * 2 * n);
    int8_t *f = bufs.data(), *nf = f + 2 * n, *r = nf + 2 * n,
           *nr = r + 2 * n;
    for (int64_t i = 0; i < n; i++) {
        f[i] = seq[i];
        r[i] = seq[n - 1 - i];
        nf[i] = (int8_t)(-seq[i]);
        nr[i] = (int8_t)(-seq[n - 1 - i]);
    }
    std::memcpy(f + n, f, n);
    std::memcpy(nf + n, nf, n);
    std::memcpy(r + n, r, n);
    std::memcpy(nr + n, nr, n);
    const int8_t *bases[4] = {f, nf, r, nr};
    const int8_t *best = f;
    const int64_t rots = periodic ? n : 1;
    for (int v = 0; v < 4; v++)
        for (int64_t rot = 0; rot < rots; rot++) {
            const int8_t *cand = bases[v] + rot;
            if (std::memcmp(cand, best, n) < 0)
                best = cand;
        }
    std::memcpy(out, best, n);
    return 0;
}

// v4: identical matching semantics to v3, plus collision-evidence capture.
// Every hash-equal probe event (memcmp pass OR fail) is recorded as
// (i, j, k, l) into coll_out (capacity coll_cap quadruples); *coll_n is the
// count. Pure observability: results and counters match v3 exactly.
extern "C" int64_t match_pairs_v4(
    const int16_t *pafA, int64_t na, const int16_t *pafB, int64_t nb,
    const int16_t *pafC, int64_t nc, const int16_t *pafD, int64_t nd,
    int64_t h, int16_t wt1, int16_t wt2,
    const float *psdA, const int32_t *wA, const float *psdB,
    const float *psdC, const int32_t *wC, const float *psdD,
    int64_t F, float cap1, float cap2,
    int64_t *out_idx, int64_t *stats,
    int64_t *coll_out, int64_t coll_cap, int64_t *coll_n) {

    for (int i = 0; i < 5; i++) stats[i] = 0;
    *coll_n = 0;
    if (na * nb == 0 || nc * nd == 0) return 0;
    const bool s1 = psdA && psdB && wA && cap1 > 0;
    const bool s2 = psdC && psdD && wC && cap2 > 0;

    uint64_t cap = 1;
    while ((int64_t)cap < 2 * na * nb) cap <<= 1;
    std::vector<uint64_t> keys(cap, 0);
    std::vector<uint32_t> ai(cap), bj(cap);
    std::vector<int16_t> buf(h);

    for (int64_t i = 0; i < na; i++) {
        const int16_t *ra = pafA + i * h;
        const float ax = s1 ? psdA[i * F + wA[i]] : 0.f;
        for (int64_t j = 0; j < nb; j++) {
            if (s1 && ax + psdB[j * F + wA[i]] > cap1 + 1e-6f) {
                stats[4]++; continue;
            }
            combine_rows(ra, pafB + j * h, wt1, buf.data(), h);
            uint64_t k = hash64w(buf.data(), h), pos = k & (cap - 1);
            while (keys[pos] != 0) pos = (pos + 1) & (cap - 1);
            keys[pos] = k; ai[pos] = (uint32_t)i; bj[pos] = (uint32_t)j;
            stats[0]++;
        }
    }
    stats[1] = stats[0];

    std::vector<int16_t> probe(h), chk(h);
    for (int64_t k2 = 0; k2 < nc; k2++) {
        const int16_t *rc = pafC + k2 * h;
        const float cx = s2 ? psdC[k2 * F + wC[k2]] : 0.f;
        for (int64_t l = 0; l < nd; l++) {
            if (s2 && cx + psdD[l * F + wC[k2]] > cap2 + 1e-6f) {
                stats[4]++; continue;
            }
            combine_rows(rc, pafD + l * h, (int16_t)(-wt2),
                         probe.data(), h);
            stats[2]++;
            uint64_t k = hash64w(probe.data(), h), pos = k & (cap - 1);
            while (keys[pos] != 0) {
                if (keys[pos] == k) {
                    stats[3]++;
                    int64_t i = ai[pos], j = bj[pos];
                    if (*coll_n < coll_cap) {
                        int64_t *rec = coll_out + (*coll_n) * 4;
                        rec[0] = i; rec[1] = j; rec[2] = k2; rec[3] = l;
                        (*coll_n)++;
                    }
                    combine_rows(pafA + i * h, pafB + j * h, wt1,
                                 chk.data(), h);
                    if (std::memcmp(chk.data(), probe.data(), h * 2) == 0) {
                        out_idx[0] = i; out_idx[1] = j;
                        out_idx[2] = k2; out_idx[3] = l;
                        return 1;
                    }
                }
                pos = (pos + 1) & (cap - 1);
            }
        }
    }
    return 0;
}

// v5: multi-frequency PSD pair screen. Instead of testing only each
// candidate's single peak frequency, test its top-K frequencies AND the
// partner's top-K (2K lookups max, early exit). Soundness is unchanged:
// an exact match satisfies PSD_x(w)+PSD_y(w) <= cap at EVERY frequency,
// so rejecting on any per-frequency violation can never lose a match.
// K=1 with one-sided check reproduces v4 screening semantics.
static inline bool psd_reject(const float *px, const float *py,
                              const int32_t *wx, const int32_t *wy,
                              int64_t x, int64_t y, int64_t F,
                              int64_t K, float cap) {
    const int32_t *ax = wx + x * K;
    for (int64_t t = 0; t < K; t++) {
        int32_t w = ax[t];
        if (w < 0) break;
        if (px[x * F + w] + py[y * F + w] > cap + 1e-6f) return true;
    }
    if (wy) {
        const int32_t *by = wy + y * K;
        for (int64_t t = 0; t < K; t++) {
            int32_t w = by[t];
            if (w < 0) break;
            if (px[x * F + w] + py[y * F + w] > cap + 1e-6f) return true;
        }
    }
    return false;
}

extern "C" int64_t match_pairs_v5(
    const int16_t *pafA, int64_t na, const int16_t *pafB, int64_t nb,
    const int16_t *pafC, int64_t nc, const int16_t *pafD, int64_t nd,
    int64_t h, int16_t wt1, int16_t wt2,
    const float *psdA, const int32_t *wA, const float *psdB,
    const int32_t *wB,
    const float *psdC, const int32_t *wC, const float *psdD,
    const int32_t *wD,
    int64_t F, int64_t K, float cap1, float cap2,
    int64_t *out_idx, int64_t *stats,
    int64_t *coll_out, int64_t coll_cap, int64_t *coll_n) {

    for (int i = 0; i < 5; i++) stats[i] = 0;
    *coll_n = 0;
    if (na * nb == 0 || nc * nd == 0) return 0;
    const bool s1 = psdA && psdB && wA && cap1 > 0 && K > 0;
    const bool s2 = psdC && psdD && wC && cap2 > 0 && K > 0;

    uint64_t cap = 1;
    while ((int64_t)cap < 2 * na * nb) cap <<= 1;
    std::vector<uint64_t> keys(cap, 0);
    std::vector<uint32_t> ai(cap), bj(cap);
    std::vector<int16_t> buf(h);

    for (int64_t i = 0; i < na; i++) {
        const int16_t *ra = pafA + i * h;
        for (int64_t j = 0; j < nb; j++) {
            if (s1 && psd_reject(psdA, psdB, wA, wB, i, j, F, K, cap1)) {
                stats[4]++; continue;
            }
            combine_rows(ra, pafB + j * h, wt1, buf.data(), h);
            uint64_t k = hash64w(buf.data(), h), pos = k & (cap - 1);
            while (keys[pos] != 0) pos = (pos + 1) & (cap - 1);
            keys[pos] = k; ai[pos] = (uint32_t)i; bj[pos] = (uint32_t)j;
            stats[0]++;
        }
    }
    stats[1] = stats[0];

    std::vector<int16_t> probe(h), chk(h);
    for (int64_t k2 = 0; k2 < nc; k2++) {
        const int16_t *rc = pafC + k2 * h;
        for (int64_t l = 0; l < nd; l++) {
            if (s2 && psd_reject(psdC, psdD, wC, wD, k2, l, F, K, cap2)) {
                stats[4]++; continue;
            }
            combine_rows(rc, pafD + l * h, (int16_t)(-wt2),
                         probe.data(), h);
            stats[2]++;
            uint64_t k = hash64w(probe.data(), h), pos = k & (cap - 1);
            while (keys[pos] != 0) {
                if (keys[pos] == k) {
                    stats[3]++;
                    int64_t i = ai[pos], j = bj[pos];
                    if (*coll_n < coll_cap) {
                        int64_t *rec = coll_out + (*coll_n) * 4;
                        rec[0] = i; rec[1] = j; rec[2] = k2; rec[3] = l;
                        (*coll_n)++;
                    }
                    combine_rows(pafA + i * h, pafB + j * h, wt1,
                                 chk.data(), h);
                    if (std::memcmp(chk.data(), probe.data(), h * 2) == 0) {
                        out_idx[0] = i; out_idx[1] = j;
                        out_idx[2] = k2; out_idx[3] = l;
                        return 1;
                    }
                }
                pos = (pos + 1) & (cap - 1);
            }
        }
    }
    return 0;
}

// v6: v5 plus same-bin swap deduplication. When two sides of a pair draw
// from the SAME bin, global-index pairs with i > j are skipped (tri flag,
// offsets map chunk-local to global indices). Soundness: the matching
// condition is symmetric in the two sequences of a side (sum of equally
// weighted autocorrelations), so if (i,j,...) is a solution so is
// (j,i,...); keeping i <= j retains a representative of every solution.
// stats[6] = { pairs, buckets, probes, collchk, psd_skip, swap_skip_build,
//              swap_skip_probe }  (first five identical to v5 semantics)
extern "C" int64_t match_pairs_v6(
    const int16_t *pafA, int64_t na, const int16_t *pafB, int64_t nb,
    const int16_t *pafC, int64_t nc, const int16_t *pafD, int64_t nd,
    int64_t h, int16_t wt1, int16_t wt2,
    const float *psdA, const int32_t *wA, const float *psdB,
    const int32_t *wB,
    const float *psdC, const int32_t *wC, const float *psdD,
    const int32_t *wD,
    int64_t F, int64_t K, float cap1, float cap2,
    int64_t a_off, int64_t b_off, int64_t c_off, int64_t d_off,
    int32_t tri1, int32_t tri2,
    int64_t *out_idx, int64_t *stats,
    int64_t *coll_out, int64_t coll_cap, int64_t *coll_n) {

    for (int i = 0; i < 7; i++) stats[i] = 0;
    *coll_n = 0;
    if (na * nb == 0 || nc * nd == 0) return 0;
    const bool s1 = psdA && psdB && wA && cap1 > 0 && K > 0;
    const bool s2 = psdC && psdD && wC && cap2 > 0 && K > 0;

    uint64_t cap = 1;
    while ((int64_t)cap < 2 * na * nb) cap <<= 1;
    std::vector<uint64_t> keys(cap, 0);
    std::vector<uint32_t> ai(cap), bj(cap);
    std::vector<int16_t> buf(h);

    for (int64_t i = 0; i < na; i++) {
        const int16_t *ra = pafA + i * h;
        for (int64_t j = 0; j < nb; j++) {
            if (tri1 && (a_off + i) > (b_off + j)) { stats[5]++; continue; }
            if (s1 && psd_reject(psdA, psdB, wA, wB, i, j, F, K, cap1)) {
                stats[4]++; continue;
            }
            combine_rows(ra, pafB + j * h, wt1, buf.data(), h);
            uint64_t k = hash64w(buf.data(), h), pos = k & (cap - 1);
            while (keys[pos] != 0) pos = (pos + 1) & (cap - 1);
            keys[pos] = k; ai[pos] = (uint32_t)i; bj[pos] = (uint32_t)j;
            stats[0]++;
        }
    }
    stats[1] = stats[0];

    std::vector<int16_t> probe(h), chk(h);
    for (int64_t k2 = 0; k2 < nc; k2++) {
        const int16_t *rc = pafC + k2 * h;
        for (int64_t l = 0; l < nd; l++) {
            if (tri2 && (c_off + k2) > (d_off + l)) { stats[6]++; continue; }
            if (s2 && psd_reject(psdC, psdD, wC, wD, k2, l, F, K, cap2)) {
                stats[4]++; continue;
            }
            combine_rows(rc, pafD + l * h, (int16_t)(-wt2),
                         probe.data(), h);
            stats[2]++;
            uint64_t k = hash64w(probe.data(), h), pos = k & (cap - 1);
            while (keys[pos] != 0) {
                if (keys[pos] == k) {
                    stats[3]++;
                    int64_t i = ai[pos], j = bj[pos];
                    if (*coll_n < coll_cap) {
                        int64_t *rec = coll_out + (*coll_n) * 4;
                        rec[0] = i; rec[1] = j; rec[2] = k2; rec[3] = l;
                        (*coll_n)++;
                    }
                    combine_rows(pafA + i * h, pafB + j * h, wt1,
                                 chk.data(), h);
                    if (std::memcmp(chk.data(), probe.data(), h * 2) == 0) {
                        out_idx[0] = i; out_idx[1] = j;
                        out_idx[2] = k2; out_idx[3] = l;
                        return 1;
                    }
                }
                pos = (pos + 1) & (cap - 1);
            }
        }
    }
    return 0;
}

// v5: multi-frequency PSD pair screen. Instead of testing only each
// candidate's single peak frequency, test its top-K frequencies AND the
// partner's top-K (2K lookups max, early exit). Soundness is unchanged:
// an exact match satisfies PSD_x(w)+PSD_y(w) <= cap at EVERY frequency,
// so rejecting on any per-frequency violation can never lose a match.
// K=1 with one-sided check reproduces v4 screening semantics.

// duplicate static inline bool psd_reject( block removed


// duplicate extern "C" int64_t match_pairs_v5( block removed

// H668 manual append: match_pairs_v7 from h668_matchspeed.patch

// v7: segmented, index-addressed matching. One hash table is built from
// nBuildSegs (idxA,idxB) row-index segment products, then probed by
// nProbeSegs (idxC,idxD) segment products. Rows are addressed through the
// idx arrays into the SAME base paf/psd arrays, so a partition of the
// pair space by an exact necessary condition (equal weighted coordinate
// sums) costs one call per partition with NO reprobe multiplication.
// Semantics per examined pair are byte-identical to v6: same combine,
// same hash, same PSD screen, same triangle rule (on idx+off global
// order), same memcmp verification, same collision capture.
extern "C" int64_t match_pairs_v7(
    const int16_t *pafA, const int16_t *pafB,
    const int16_t *pafC, const int16_t *pafD, int64_t h,
    int16_t wt1, int16_t wt2,
    const float *psdA, const int32_t *wA, const float *psdB,
    const int32_t *wB,
    const float *psdC, const int32_t *wC, const float *psdD,
    const int32_t *wD,
    int64_t F, int64_t K, float cap1, float cap2,
    const int64_t *bIdxA, const int64_t *bIdxB,
    const int64_t *bStartA, const int64_t *bStartB, int64_t nBuild,
    const int64_t *pIdxC, const int64_t *pIdxD,
    const int64_t *pStartC, const int64_t *pStartD, int64_t nProbe,
    int64_t offA, int64_t offB, int64_t offC, int64_t offD,
    int32_t tri1, int32_t tri2, int64_t table_pairs,
    int64_t *out_idx, int64_t *stats,
    int64_t *coll_out, int64_t coll_cap, int64_t *coll_n) {

    for (int i = 0; i < 7; i++) stats[i] = 0;
    *coll_n = 0;
    if (nBuild == 0 || nProbe == 0) return 0;
    const bool s1 = psdA && psdB && wA && cap1 > 0 && K > 0;
    const bool s2 = psdC && psdD && wC && cap2 > 0 && K > 0;

    uint64_t cap = 16;
    while ((int64_t)cap < 2 * table_pairs) cap <<= 1;
    std::vector<uint64_t> keys(cap, 0);
    std::vector<int64_t> ai(cap), bj(cap);
    std::vector<int16_t> buf(h);

    for (int64_t s = 0; s < nBuild; s++) {
        for (int64_t x = bStartA[s]; x < bStartA[s + 1]; x++) {
            const int64_t i = bIdxA[x];
            const int16_t *ra = pafA + i * h;
            for (int64_t y = bStartB[s]; y < bStartB[s + 1]; y++) {
                const int64_t j = bIdxB[y];
                if (tri1 && (offA + i) > (offB + j)) {
                    stats[5]++; continue;
                }
                if (s1 && psd_reject(psdA, psdB, wA, wB, i, j, F, K,
                                     cap1)) {
                    stats[4]++; continue;
                }
                combine_rows(ra, pafB + j * h, wt1, buf.data(), h);
                uint64_t k = hash64w(buf.data(), h),
                         pos = k & (cap - 1);
                while (keys[pos] != 0) pos = (pos + 1) & (cap - 1);
                keys[pos] = k; ai[pos] = i; bj[pos] = j;
                stats[0]++;
            }
        }
    }
    stats[1] = stats[0];

    std::vector<int16_t> probe(h), chk(h);
    for (int64_t s = 0; s < nProbe; s++) {
        for (int64_t x = pStartC[s]; x < pStartC[s + 1]; x++) {
            const int64_t k2 = pIdxC[x];
            const int16_t *rc = pafC + k2 * h;
            for (int64_t y = pStartD[s]; y < pStartD[s + 1]; y++) {
                const int64_t l = pIdxD[y];
                if (tri2 && (offC + k2) > (offD + l)) {
                    stats[6]++; continue;
                }
                if (s2 && psd_reject(psdC, psdD, wC, wD, k2, l, F, K,
                                     cap2)) {
                    stats[4]++; continue;
                }
                combine_rows(rc, pafD + l * h, (int16_t)(-wt2),
                             probe.data(), h);
                stats[2]++;
                uint64_t k = hash64w(probe.data(), h),
                         pos = k & (cap - 1);
                while (keys[pos] != 0) {
                    if (keys[pos] == k) {
                        stats[3]++;
                        int64_t i = ai[pos], j = bj[pos];
                        if (*coll_n < coll_cap) {
                            int64_t *rec = coll_out + (*coll_n) * 4;
                            rec[0] = i; rec[1] = j;
                            rec[2] = k2; rec[3] = l;
                            (*coll_n)++;
                        }
                        combine_rows(pafA + i * h, pafB + j * h, wt1,
                                     chk.data(), h);
                        if (std::memcmp(chk.data(), probe.data(),
                                        h * 2) == 0) {
                            out_idx[0] = i; out_idx[1] = j;
                            out_idx[2] = k2; out_idx[3] = l;
                            return 1;
                        }
                    }
                    pos = (pos + 1) & (cap - 1);
                }
            }
        }
    }
    return 0;
}
