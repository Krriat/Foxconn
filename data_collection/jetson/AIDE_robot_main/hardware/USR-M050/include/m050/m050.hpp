#ifndef M050_M050_HPP
#define M050_M050_HPP

#include "m050.h"
#include <memory>
#include <stdexcept>
#include <string>

namespace m050 {

class Error : public std::runtime_error {
public:
    explicit Error(const m050_error &error)
        : std::runtime_error(error.message[0] ? error.message : m050_result_string(error.code)),
          detail_(error) {}
    int code() const noexcept { return detail_.code; }
    bool write_may_have_applied() const noexcept { return detail_.write_may_have_applied != 0; }
    const m050_error &detail() const noexcept { return detail_; }
private:
    m050_error detail_;
};

struct Channels { bool ch1; bool ch2; };
struct IOState { Channels di; Channels do_state; };

class Client {
public:
    explicit Client(const std::string &host = M050_DEFAULT_HOST,
                    unsigned port = M050_DEFAULT_PORT,
                    unsigned unit_id = M050_DEFAULT_UNIT_ID,
                    int timeout_ms = M050_DEFAULT_TIMEOUT_MS,
                    const std::string &source_ip = "")
        : client_(nullptr, &m050_destroy)
    {
        if (!port || port > 65535 || !unit_id || unit_id > 247)
            throw std::invalid_argument("invalid port or unit ID");
        m050_config config;
        m050_config_default(&config);
        config.host = host.c_str();
        config.port = static_cast<uint16_t>(port);
        config.unit_id = static_cast<uint8_t>(unit_id);
        config.timeout_ms = timeout_ms;
        config.source_ip = source_ip.empty() ? nullptr : source_ip.c_str();
        m050_error error;
        m050_client *raw = nullptr;
        check(m050_create(&config, &raw, &error), error);
        client_.reset(raw);
    }
    Client(const Client &) = delete;
    Client &operator=(const Client &) = delete;
    Client(Client &&) noexcept = default;
    Client &operator=(Client &&) noexcept = default;

    Channels read_do() {
        m050_channels state;
        m050_error error;
        check(m050_read_do(client_.get(), &state, &error), error);
        return convert(state);
    }
    Channels read_di() {
        m050_channels state;
        m050_error error;
        check(m050_read_di(client_.get(), &state, &error), error);
        return convert(state);
    }
    IOState read_io() {
        m050_io_state state;
        m050_error error;
        check(m050_read_io(client_.get(), &state, &error), error);
        return {convert(state.di), convert(state.do_state)};
    }
    Channels set_do(unsigned channel, bool closed) {
        m050_channels state;
        m050_error error;
        check(m050_set_do(client_.get(), channel, closed ? 1 : 0, &state, &error), error);
        return convert(state);
    }
private:
    static void check(int result, const m050_error &error) {
        if (result != M050_OK) throw Error(error);
    }
    static Channels convert(const m050_channels &state) {
        return {state.ch1 != 0, state.ch2 != 0};
    }
    std::unique_ptr<m050_client, void (*)(m050_client *)> client_;
};

} // namespace m050
#endif
