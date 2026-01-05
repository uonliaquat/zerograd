
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define SEED     0x12345678

uint32_t MurmurOAAT_32(const char* str, uint32_t h)
{
    // One-byte-at-a-time hash based on Murmur's mix
    // Source: https://github.com/aappleby/smhasher/blob/master/src/Hashes.cpp
    for (; *str; ++str) {
        h ^= *str;
        h *= 0x5bd1e995;
        h ^= h >> 15;
    }
    h &= 0xFFFF;   // keep only 16 bits
    return h;
}

int main(){
    char data[10] ="hell\0";
    uint32_t hash = MurmurOAAT_32(data, SEED);
    printf("hash: %u\n", hash);
    return 0;
}