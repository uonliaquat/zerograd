/* safetensors_loader.c — minimal random-access loader.
 *
 * Format:
 *   [8 bytes]            little-endian u64 = header_len (N)
 *   [N bytes]            JSON header: { "name": {"dtype","shape","data_offsets":[a,b]}, ... }
 *   [rest]               raw tensor bytes; data_offsets a,b are relative to the
 *                        START of this block (i.e. file offset 8 + N + a).
 *
 * Usage:
 *   SafeTensors st;
 *   st_open(&st, "/path/gpt2_split.safetensors");
 *   st_load(&st, "h.0.q.weight", ctx->mem + tensor->data_offset, tensor->nbytes);
 *   st_close(&st);
 */

 #include "../../inc/loader/loader_safetensors.h"
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>


int st_open(SafeTensors *st, const char *path) {
    memset(st, 0, sizeof(*st));
    st->fd = open(path, O_RDONLY);
    if (st->fd < 0) { perror("open"); return -1; }

    struct stat sb;
    if (fstat(st->fd, &sb) != 0) { perror("fstat"); close(st->fd); return -1; }
    st->file_size = (size_t)sb.st_size;

    st->base = mmap(NULL, st->file_size, PROT_READ, MAP_PRIVATE, st->fd, 0);
    if (st->base == MAP_FAILED) { perror("mmap"); close(st->fd); return -1; }

    uint64_t n;
    memcpy(&n, st->base, 8);          /* header length, little-endian */
    st->json     = (char *)(st->base + 8);
    st->json_len = (size_t)n;
    st->data     = st->base + 8 + (size_t)n;
    return 0;
}

void st_close(SafeTensors *st) {
    if (st->base && st->base != MAP_FAILED) munmap(st->base, st->file_size);
    if (st->fd >= 0) close(st->fd);
    memset(st, 0, sizeof(*st));
}

/* Find a tensor's data_offsets [begin, end] within the data block.
 * Returns 0 on success. Searches for the quoted key "<name>": then the
 * following "data_offsets":[a,b]. */
int st_find(SafeTensors *st, const char *name,
            size_t *begin, size_t *end) {
    /* build the search needle:  "name":   */
    char needle[256];
    int len = snprintf(needle, sizeof(needle), "\"%s\":", name);
    if (len < 0 || (size_t)len >= sizeof(needle)) return -1;

    /* json is not NUL-terminated; bound the search with memmem-style scan */
    char *p = NULL;
    for (size_t i = 0; i + (size_t)len <= st->json_len; i++) {
        if (memcmp(st->json + i, needle, (size_t)len) == 0) {
            p = st->json + i;
            break;
        }
    }
    if (!p) { fprintf(stderr, "tensor not found: %s\n", name); return -1; }

    const char *key = "\"data_offsets\":[";
    char *q = strstr(p, key);           /* header has a NUL after it in practice;
                                           if not, scan as above. mmap'd files
                                           are usually padded — see note below. */
    if (!q) {
        /* manual bounded scan as a fallback */
        size_t klen = strlen(key);
        for (size_t i = (size_t)(p - st->json);
             i + klen <= st->json_len; i++) {
            if (memcmp(st->json + i, key, klen) == 0) { q = st->json + i; break; }
        }
        if (!q) return -1;
    }
    q += strlen(key);

    char *endp;
    unsigned long long a = strtoull(q, &endp, 10);
    while (*endp == ',' || *endp == ' ') endp++;
    unsigned long long b = strtoull(endp, &endp, 10);

    *begin = (size_t)a;
    *end   = (size_t)b;
    return 0;
}

/* Copy a named tensor's raw bytes into dst.
 * expected_nbytes: pass the destination tensor's byte size for a safety check
 * (or 0 to skip). Returns number of bytes copied, or 0 on error. */
size_t st_load(SafeTensors *st, const char *name,
               void *dst, size_t expected_nbytes) {
    size_t begin, end;
    if (st_find(st, name, &begin, &end) != 0) return 0;

    size_t nbytes = end - begin;
    if (expected_nbytes && nbytes != expected_nbytes) {
        fprintf(stderr,
            "size mismatch for %s: file has %zu bytes, dst expects %zu\n",
            name, nbytes, expected_nbytes);
        return 0;
    }
    if (st->data + end > st->base + st->file_size) {
        fprintf(stderr, "tensor %s out of file bounds\n", name);
        return 0;
    }

    memcpy(dst, st->data + begin, nbytes);
    return nbytes;
}

/* Zero-copy variant: return a pointer into the mmap'd data instead of copying.
 * Valid until st_close. Useful if you can point your tensor at it directly. */
const void *st_view(SafeTensors *st, const char *name, size_t *nbytes_out) {
    size_t begin, end;
    if (st_find(st, name, &begin, &end) != 0) return NULL;
    if (nbytes_out) *nbytes_out = end - begin;
    return st->data + begin;
}


#ifdef ST_DEMO
int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s file tensor_name\n", argv[0]); return 1; }
    SafeTensors st;
    if (st_open(&st, argv[1]) != 0) return 1;

    size_t nb;
    const float *v = (const float *)st_view(&st, argv[2], &nb);
    if (v) {
        printf("%s: %zu bytes (%zu floats)\n", argv[2], nb, nb / sizeof(float));
        printf("first 8: ");
        for (int i = 0; i < 8 && (size_t)i < nb / sizeof(float); i++)
            printf("%.4f ", v[i]);
        printf("\n");
    }
    st_close(&st);
    return 0;
}
#endif