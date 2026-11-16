#include <m050/m050.hpp>
#include <getopt.h>
#include <iostream>
#include <string>

static void usage(std::ostream &out)
{
    out << "Usage: m050ctl [options] [status|read-do|read-di|set-do CHANNEL on|off]\n"
           "  --host IPv4      default: 192.168.1.7\n"
           "  --port PORT      default: 20108\n"
           "  --unit ID        default: 1 (1..247)\n"
           "  --timeout MS     whole network operation; default: 3000\n"
           "  --source IPv4    optional local source address\n"
           "  --help           show this help\n"
           "Default command is status (read only). set-do writes one relay and verifies it.\n";
}

static unsigned number(const char *text, unsigned minimum, unsigned maximum)
{
    std::string value(text);
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("expected an unsigned decimal number: " + value);
    std::size_t end;
    unsigned long parsed = std::stoul(value, &end, 10);
    if (end != value.size() || parsed < minimum || parsed > maximum)
        throw std::invalid_argument("numeric argument outside allowed range: " + value);
    return static_cast<unsigned>(parsed);
}

static void print_channels(const char *name, const m050::Channels &state)
{
    std::cout << name << "1=" << state.ch1 << ' ' << name << "2=" << state.ch2 << '\n';
}

int main(int argc, char **argv)
{
    std::string host = M050_DEFAULT_HOST, source;
    unsigned port = M050_DEFAULT_PORT, unit = M050_DEFAULT_UNIT_ID;
    int timeout = M050_DEFAULT_TIMEOUT_MS;
    const option options[] = {
        {"host", required_argument, nullptr, 'H'},
        {"port", required_argument, nullptr, 'p'},
        {"unit", required_argument, nullptr, 'u'},
        {"timeout", required_argument, nullptr, 't'},
        {"source", required_argument, nullptr, 's'},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0}
    };
    try {
        int opt;
        while ((opt = getopt_long(argc, argv, "+H:p:u:t:s:h", options, nullptr)) != -1) {
            switch (opt) {
            case 'H': host = optarg; break;
            case 'p': port = number(optarg, 1, 65535); break;
            case 'u': unit = number(optarg, 1, 247); break;
            case 't': timeout = static_cast<int>(number(optarg, 1, 2147483647)); break;
            case 's': source = optarg; break;
            case 'h': usage(std::cout); return 0;
            default: usage(std::cerr); return 2;
            }
        }
        std::string command = optind < argc ? argv[optind++] : "status";
        unsigned channel = 0;
        bool closed = false;
        if (command == "set-do") {
            if (argc - optind != 2) throw std::invalid_argument("set-do needs CHANNEL and on/off");
            channel = number(argv[optind++], 1, 2);
            std::string state = argv[optind++];
            if (state == "on" || state == "1") closed = true;
            else if (state == "off" || state == "0") closed = false;
            else throw std::invalid_argument("DO state must be on/off or 1/0");
        } else if (command != "status" && command != "read-do" && command != "read-di") {
            throw std::invalid_argument("unknown command: " + command);
        }
        if (optind != argc) throw std::invalid_argument("unexpected trailing arguments");
        m050::Client client(host, port, unit, timeout, source);
        if (command == "status") {
            const auto state = client.read_io();
            print_channels("DO", state.do_state);
            print_channels("DI", state.di);
        } else if (command == "read-do") {
            print_channels("DO", client.read_do());
        } else if (command == "read-di") {
            print_channels("DI", client.read_di());
        } else {
            print_channels("DO", client.set_do(channel, closed));
        }
        return 0;
    } catch (const m050::Error &error) {
        std::cerr << "M050 error " << error.code() << ": " << error.what() << '\n';
        if (error.write_may_have_applied())
            std::cerr << "Write may already have applied. Read DO status before any retry.\n";
        return 1;
    } catch (const std::exception &error) {
        std::cerr << "Argument error: " << error.what() << '\n';
        usage(std::cerr);
        return 2;
    }
}
