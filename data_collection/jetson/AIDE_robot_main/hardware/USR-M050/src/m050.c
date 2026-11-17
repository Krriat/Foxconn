#define _POSIX_C_SOURCE 200809L
#include "m050/m050.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

struct m050_client {
    struct sockaddr_in remote;
    struct sockaddr_in source;
    int bind_source;
    int timeout_ms;
    uint8_t unit_id;
    uint16_t transaction_id;
    int reuse_connection;
    int retained_fd;
    pthread_mutex_t mutex;
};

typedef struct {
    m050_client *client;
    m050_error *error;
    int fd;
    int64_t deadline_ms;
    int write_sent;
} operation;

static void clear_error(m050_error *error)
{
    if (error) memset(error, 0, sizeof(*error));
}

static int fail(m050_error *error, int code, int system_errno,
                const char *format, ...)
{
    if (error) {
        va_list args;
        error->code = code;
        error->system_errno = system_errno;
        va_start(args, format);
        vsnprintf(error->message, sizeof(error->message), format, args);
        va_end(args);
    }
    return code;
}

void m050_config_default(m050_config *config)
{
    if (!config) return;
    config->host = M050_DEFAULT_HOST;
    config->port = M050_DEFAULT_PORT;
    config->unit_id = M050_DEFAULT_UNIT_ID;
    config->timeout_ms = M050_DEFAULT_TIMEOUT_MS;
    config->source_ip = NULL;
}

const char *m050_result_string(int result)
{
    switch (result) {
    case M050_OK: return "success";
    case M050_ERR_ARGUMENT: return "invalid argument";
    case M050_ERR_MEMORY: return "out of memory";
    case M050_ERR_SYSTEM: return "system error";
    case M050_ERR_CONNECT: return "connection failed";
    case M050_ERR_TIMEOUT: return "operation timed out";
    case M050_ERR_IO: return "socket I/O failed";
    case M050_ERR_PROTOCOL: return "invalid Modbus TCP response";
    case M050_ERR_MODBUS: return "Modbus exception";
    case M050_ERR_VERIFY: return "DO readback mismatch";
    default: return "unknown error";
    }
}

int m050_create(const m050_config *config, m050_client **out, m050_error *error)
{
    m050_config defaults;
    m050_client *client;
    struct sockaddr_in remote = {0}, source = {0};
    int rc;
    clear_error(error);
    if (!out) return fail(error, M050_ERR_ARGUMENT, 0, "out must not be NULL");
    *out = NULL;
    if (!config) {
        m050_config_default(&defaults);
        config = &defaults;
    }
    if (!config->host || !config->port || !config->unit_id ||
        config->unit_id > 247 || config->timeout_ms <= 0)
        return fail(error, M050_ERR_ARGUMENT, 0, "invalid endpoint, unit ID, or timeout");
    remote.sin_family = AF_INET;
    remote.sin_port = htons(config->port);
    if (inet_pton(AF_INET, config->host, &remote.sin_addr) != 1)
        return fail(error, M050_ERR_ARGUMENT, 0, "host must be a numeric IPv4 address");
    source.sin_family = AF_INET;
    if (config->source_ip &&
        inet_pton(AF_INET, config->source_ip, &source.sin_addr) != 1)
        return fail(error, M050_ERR_ARGUMENT, 0, "source_ip must be a numeric IPv4 address");
    client = calloc(1, sizeof(*client));
    if (!client) return fail(error, M050_ERR_MEMORY, errno, "allocating client failed");
    client->remote = remote;
    client->source = source;
    client->bind_source = config->source_ip != NULL;
    client->timeout_ms = config->timeout_ms;
    client->unit_id = config->unit_id;
    client->retained_fd = -1;
    rc = pthread_mutex_init(&client->mutex, NULL);
    if (rc) {
        free(client);
        return fail(error, M050_ERR_SYSTEM, rc, "initializing mutex: %s", strerror(rc));
    }
    *out = client;
    return M050_OK;
}

void m050_destroy(m050_client *client)
{
    if (!client) return;
    if (client->retained_fd >= 0) close(client->retained_fd);
    pthread_mutex_destroy(&client->mutex);
    free(client);
}

static int64_t monotonic_ms(void)
{
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return -1;
    return (int64_t)now.tv_sec * 1000 + now.tv_nsec / 1000000;
}

static int wait_socket(operation *op, short events)
{
    struct pollfd pfd = {op->fd, events, 0};
    for (;;) {
        int rc;
        int64_t now = monotonic_ms();
        int64_t remaining;
        if (now < 0)
            return fail(op->error, M050_ERR_SYSTEM, errno, "reading monotonic clock failed");
        remaining = op->deadline_ms - now;
        if (remaining <= 0)
            return fail(op->error, M050_ERR_TIMEOUT, 0, "network operation exceeded %d ms",
                        op->client->timeout_ms);
        rc = poll(&pfd, 1, remaining > INT_MAX ? INT_MAX : (int)remaining);
        if (rc < 0 && errno == EINTR) continue;
        if (rc < 0)
            return fail(op->error, M050_ERR_IO, errno, "poll: %s", strerror(errno));
        if (rc == 0) continue;
        if (pfd.revents & POLLNVAL)
            return fail(op->error, M050_ERR_IO, EBADF, "invalid socket descriptor");
        /* Let send/recv/getsockopt report EOF and socket errors, including HUP. */
        if (pfd.revents & (events | POLLERR | POLLHUP)) return M050_OK;
    }
}

static int begin_operation(operation *op)
{
    int rc, socket_error;
    socklen_t error_size = sizeof(socket_error);
    int64_t now = monotonic_ms();
    if (now < 0)
        return fail(op->error, M050_ERR_SYSTEM, errno, "reading monotonic clock failed");
    op->deadline_ms = now + op->client->timeout_ms;
    if (op->client->retained_fd >= 0) {
        struct pollfd retained = {op->client->retained_fd, POLLIN, 0};
        op->fd = op->client->retained_fd;
        op->client->retained_fd = -1;
        /* Reconnect before sending if an idle peer closed, or unsolicited data
           remains. A request that has been sent is never replayed. */
        if (poll(&retained, 1, 0) == 0) return M050_OK;
        close(op->fd);
        op->fd = -1;
    }
    op->fd = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (op->fd < 0)
        return fail(op->error, M050_ERR_SYSTEM, errno, "socket: %s", strerror(errno));
    if (op->client->reuse_connection) {
        const int enabled = 1;
        if (setsockopt(op->fd, IPPROTO_TCP, TCP_NODELAY, &enabled, sizeof(enabled)) < 0)
            return fail(op->error, M050_ERR_SYSTEM, errno, "TCP_NODELAY: %s", strerror(errno));
    }
    if (op->client->bind_source &&
        bind(op->fd, (struct sockaddr *)&op->client->source, sizeof(op->client->source)) < 0)
        return fail(op->error, M050_ERR_CONNECT, errno, "binding source IPv4: %s", strerror(errno));
    rc = connect(op->fd, (struct sockaddr *)&op->client->remote, sizeof(op->client->remote));
    if (rc < 0 && errno != EINPROGRESS && errno != EINTR)
        return fail(op->error, M050_ERR_CONNECT, errno, "connect: %s", strerror(errno));
    if (rc == 0) return M050_OK;
    rc = wait_socket(op, POLLOUT);
    if (rc) return rc;
    if (getsockopt(op->fd, SOL_SOCKET, SO_ERROR, &socket_error, &error_size) < 0)
        return fail(op->error, M050_ERR_CONNECT, errno, "connect status: %s", strerror(errno));
    if (socket_error)
        return fail(op->error, M050_ERR_CONNECT, socket_error, "connect: %s", strerror(socket_error));
    return M050_OK;
}

static int send_exact(operation *op, const uint8_t *data, size_t size, int is_write)
{
    size_t offset = 0;
    while (offset < size) {
        ssize_t sent;
        int rc = wait_socket(op, POLLOUT);
        if (rc) return rc;
        sent = send(op->fd, data + offset, size - offset, MSG_NOSIGNAL);
        if (sent < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) continue;
        if (sent <= 0)
            return fail(op->error, M050_ERR_IO, sent < 0 ? errno : 0,
                        "sending Modbus request failed");
        if (is_write) op->write_sent = 1;
        offset += (size_t)sent;
    }
    return M050_OK;
}

static int recv_exact(operation *op, uint8_t *data, size_t size)
{
    size_t offset = 0;
    while (offset < size) {
        ssize_t received;
        int rc = wait_socket(op, POLLIN);
        if (rc) return rc;
        received = recv(op->fd, data + offset, size - offset, 0);
        if (received < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) continue;
        if (received <= 0)
            return fail(op->error, M050_ERR_IO, received < 0 ? errno : 0,
                        received == 0 ? "device closed connection before full response" :
                                        "receiving Modbus response failed");
        offset += (size_t)received;
    }
    return M050_OK;
}

static uint16_t read_u16(const uint8_t *p)
{
    return (uint16_t)((uint16_t)p[0] << 8 | p[1]);
}

static void write_u16(uint8_t *p, uint16_t value)
{
    p[0] = (uint8_t)(value >> 8);
    p[1] = (uint8_t)value;
}

/* All supported requests have a five-byte PDU. Max Modbus TCP PDU is 253. */
static int request(operation *op, const uint8_t pdu[5], uint8_t response[253],
                   size_t *response_size, int is_write)
{
    uint8_t packet[12], header[7];
    uint16_t tid = ++op->client->transaction_id;
    uint16_t length;
    int rc;
    write_u16(packet, tid);
    write_u16(packet + 2, 0);
    write_u16(packet + 4, 6);
    packet[6] = op->client->unit_id;
    memcpy(packet + 7, pdu, 5);
    rc = send_exact(op, packet, sizeof(packet), is_write);
    if (rc) return rc;
    rc = recv_exact(op, header, sizeof(header));
    if (rc) return rc;
    length = read_u16(header + 4);
    if (read_u16(header) != tid || read_u16(header + 2) != 0 ||
        header[6] != op->client->unit_id)
        return fail(op->error, M050_ERR_PROTOCOL, 0, "transaction/protocol/unit ID mismatch");
    if (length < 2 || length > 254)
        return fail(op->error, M050_ERR_PROTOCOL, 0, "invalid MBAP length %u", (unsigned)length);
    *response_size = length - 1;
    rc = recv_exact(op, response, *response_size);
    if (rc) return rc;
    if (response[0] == (uint8_t)(pdu[0] | 0x80)) {
        if (*response_size != 2)
            return fail(op->error, M050_ERR_PROTOCOL, 0, "malformed Modbus exception");
        if (op->error) op->error->modbus_exception = response[1];
        return fail(op->error, M050_ERR_MODBUS, 0, "Modbus exception 0x%02X for FC%02X",
                    (unsigned)response[1], (unsigned)pdu[0]);
    }
    if (response[0] != pdu[0])
        return fail(op->error, M050_ERR_PROTOCOL, 0, "response function mismatch");
    return M050_OK;
}

static int read_channels(operation *op, uint8_t function, m050_channels *state)
{
    uint8_t pdu[5] = {function, 0, 0, 0, 2}, response[253];
    size_t size;
    int rc = request(op, pdu, response, &size, 0);
    if (rc) return rc;
    if (size != 3 || response[1] != 1)
        return fail(op->error, M050_ERR_PROTOCOL, 0, "invalid two-channel status response");
    state->ch1 = response[2] & 1;
    state->ch2 = (response[2] >> 1) & 1;
    return M050_OK;
}

static int lock_client(m050_client *client, m050_error *error)
{
    int rc;
    if (!client) return fail(error, M050_ERR_ARGUMENT, 0, "client must not be NULL");
    rc = pthread_mutex_lock(&client->mutex);
    if (rc) return fail(error, M050_ERR_SYSTEM, rc, "locking client: %s", strerror(rc));
    return M050_OK;
}

static int finish_operation(operation *op, int rc)
{
    if (op->fd >= 0) {
        if (rc == M050_OK && op->client->reuse_connection)
            op->client->retained_fd = op->fd;
        else
            close(op->fd);
    }
    if (rc != M050_OK && op->error) op->error->write_may_have_applied = op->write_sent;
    pthread_mutex_unlock(&op->client->mutex);
    return rc;
}

int m050_set_connection_reuse(m050_client *client, int enabled, m050_error *error)
{
    int rc;
    clear_error(error);
    if (enabled != 0 && enabled != 1)
        return fail(error, M050_ERR_ARGUMENT, 0, "enabled must be 0 or 1");
    rc = lock_client(client, error);
    if (rc) return rc;
    client->reuse_connection = enabled;
    if (!enabled && client->retained_fd >= 0) {
        close(client->retained_fd);
        client->retained_fd = -1;
    }
    pthread_mutex_unlock(&client->mutex);
    return M050_OK;
}

static int read_one(m050_client *client, uint8_t function,
                    m050_channels *state, m050_error *error)
{
    operation op = {client, error, -1, 0, 0};
    m050_channels temporary;
    int rc;
    clear_error(error);
    if (!state) return fail(error, M050_ERR_ARGUMENT, 0, "state must not be NULL");
    rc = lock_client(client, error);
    if (rc) return rc;
    rc = begin_operation(&op);
    if (!rc) rc = read_channels(&op, function, &temporary);
    if (!rc) *state = temporary;
    return finish_operation(&op, rc);
}

int m050_read_do(m050_client *client, m050_channels *state, m050_error *error)
{
    return read_one(client, 0x01, state, error);
}

int m050_read_di(m050_client *client, m050_channels *state, m050_error *error)
{
    return read_one(client, 0x02, state, error);
}

int m050_read_io(m050_client *client, m050_io_state *state, m050_error *error)
{
    operation op = {client, error, -1, 0, 0};
    m050_io_state temporary;
    int rc;
    clear_error(error);
    if (!state) return fail(error, M050_ERR_ARGUMENT, 0, "state must not be NULL");
    rc = lock_client(client, error);
    if (rc) return rc;
    rc = begin_operation(&op);
    if (!rc) rc = read_channels(&op, 0x01, &temporary.do_state);
    if (!rc) rc = read_channels(&op, 0x02, &temporary.di);
    if (!rc) *state = temporary;
    return finish_operation(&op, rc);
}

int m050_set_do(m050_client *client, unsigned channel, int closed,
                m050_channels *verified, m050_error *error)
{
    operation op = {client, error, -1, 0, 0};
    uint8_t pdu[5] = {0x05, 0, 0, 0, 0}, response[253];
    m050_channels temporary;
    size_t size;
    int rc;
    clear_error(error);
    if ((channel != 1 && channel != 2) || (closed != 0 && closed != 1))
        return fail(error, M050_ERR_ARGUMENT, 0, "channel must be 1/2 and closed must be 0/1");
    rc = lock_client(client, error);
    if (rc) return rc;
    write_u16(pdu + 1, (uint16_t)(channel - 1));
    write_u16(pdu + 3, closed ? 0xFF00 : 0);
    rc = begin_operation(&op);
    if (!rc) rc = request(&op, pdu, response, &size, 1);
    if (!rc && (size != sizeof(pdu) || memcmp(response, pdu, sizeof(pdu)) != 0))
        rc = fail(error, M050_ERR_PROTOCOL, 0, "FC05 write echo mismatch");
    if (!rc) rc = read_channels(&op, 0x01, &temporary);
    if (!rc && (channel == 1 ? temporary.ch1 : temporary.ch2) != closed)
        rc = fail(error, M050_ERR_VERIFY, 0, "DO%u readback differs from requested %d", channel, closed);
    if (!rc && verified) *verified = temporary;
    return finish_operation(&op, rc);
}
