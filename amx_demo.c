#include <stdint.h>
#include <stdio.h>
#include <immintrin.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

// Structure representing tile config
typedef struct {
    uint8_t palette_id;
    uint8_t reserved[15];
    uint16_t colsb[8];
    uint8_t rows[8];
    uint8_t reserved2[16];
} __attribute__((packed, aligned(64))) tile_config_t;

// syscall number for arch_prctl
#ifndef ARCH_SET_STATE_ENABLE
#define ARCH_SET_STATE_ENABLE 0x1022
#endif

#define XFEATURE_AMX_TILE 18
#define AMX_TILE_MASK (1 << XFEATURE_AMX_TILE)

// Enable AMX using arch_prctl syscall
int enable_amx() {
    return syscall(SYS_arch_prctl, ARCH_SET_STATE_ENABLE, AMX_TILE_MASK);
}

// Load tile config
void load_tile_config(tile_config_t* config) {
    __tile_loadconfig(config);
}

// Clear tiles (AMX tile release)
void release_tiles() {
    __tile_release();
}

// Demo function: enable, configure, release
int run_amx_demo() {
    if (enable_amx() != 0) {
        perror("AMX enable failed");
        return -1;
    }

    tile_config_t config;
    memset(&config, 0, sizeof(config));
    config.palette_id = 1;
    config.colsb[0] = 64; // 64 bytes per row
    config.rows[0] = 16;  // 16 rows

    load_tile_config(&config);
    printf("AMX tile loaded\n");

    release_tiles();
    printf("AMX tile released\n");
    return 0;
}
