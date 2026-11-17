#ifndef M050_M050_H
#define M050_M050_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define M050_DEFAULT_HOST "192.168.1.7"
#define M050_DEFAULT_PORT 20108
#define M050_DEFAULT_UNIT_ID 1
#define M050_DEFAULT_TIMEOUT_MS 3000

typedef struct m050_client m050_client;

typedef enum {
    M050_OK = 0,
    M050_ERR_ARGUMENT = -1,
    M050_ERR_MEMORY = -2,
    M050_ERR_SYSTEM = -3,
    M050_ERR_CONNECT = -4,
    M050_ERR_TIMEOUT = -5,
    M050_ERR_IO = -6,
    M050_ERR_PROTOCOL = -7,
    M050_ERR_MODBUS = -8,
    M050_ERR_VERIFY = -9
} m050_result;

typedef struct {
    const char *host;       /* Numeric IPv4 address, copied during create. */
    uint16_t port;
    uint8_t unit_id;        /* 1..247; broadcast is deliberately unsupported. */
    int timeout_ms;         /* Whole network operation, including readback. */
    const char *source_ip;  /* Optional local numeric IPv4 address. */
} m050_config;

typedef struct {
    uint8_t ch1;            /* Always 0 or 1 on success. */
    uint8_t ch2;
} m050_channels;

typedef struct {
    m050_channels di;
    m050_channels do_state;
} m050_io_state;

typedef struct {
    int code;
    int system_errno;
    uint8_t modbus_exception;
    /* On a failed write: some command bytes were sent; do not assume OFF.
       This flag is conservative. Read status before deciding what to do next. */
    int write_may_have_applied;
    char message[192];
} m050_error;

/* All functions return M050_OK or a negative m050_result. Error is optional.
   Each call owns its error output; no global/shared last-error state exists.
   State outputs are only changed on success. */
void m050_config_default(m050_config *config);
const char *m050_result_string(int result);
int m050_create(const m050_config *config, m050_client **out, m050_error *error);
/* Destroy only after all calls using this handle have finished. NULL is valid. */
void m050_destroy(m050_client *client);

/* Optional connection reuse, disabled by default. Does not connect or write.
   Success retains the connection; errors close it, without retrying a request.
   Enabling/disabling is serialized with IO. Destroy closes a retained socket. */
int m050_set_connection_reuse(m050_client *client, int enabled, m050_error *error);

/* Calls on the same handle are serialized. Short connections are the default.
   timeout_ms starts after acquiring the handle lock; there are no retries. */
int m050_read_do(m050_client *client, m050_channels *state, m050_error *error);
int m050_read_di(m050_client *client, m050_channels *state, m050_error *error);
/* Reads DO then DI on one connection; these are sequential, not atomic. */
int m050_read_io(m050_client *client, m050_io_state *state, m050_error *error);
/* channel: 1 or 2; closed: exactly 0 or 1. FC05 followed by FC01 verification.
   verified is optional, and receives both DO states only after verification. */
int m050_set_do(m050_client *client, unsigned channel, int closed,
                m050_channels *verified, m050_error *error);

#ifdef __cplusplus
}
#endif
#endif
