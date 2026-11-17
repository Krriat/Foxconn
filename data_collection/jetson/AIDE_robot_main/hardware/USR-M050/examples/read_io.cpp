#include <m050/m050.hpp>
#include <iostream>

int main(int argc, char **argv)
{
    try {
        m050::Client io(argc > 1 ? argv[1] : M050_DEFAULT_HOST);
        const auto state = io.read_io();
        std::cout << "DO1=" << state.do_state.ch1 << " DO2=" << state.do_state.ch2
                  << " DI1=" << state.di.ch1 << " DI2=" << state.di.ch2 << '\n';
        // Call at the desired point in your business logic:
        // io.set_do(1, true);   // DO1 close
        // io.set_do(1, false);  // DO1 open
        // io.set_do(2, true);   // DO2 close
        // io.set_do(2, false);  // DO2 open
    } catch (const m050::Error &error) {
        std::cerr << error.what() << '\n';
        if (error.write_may_have_applied()) std::cerr << "Read DO before retrying.\n";
        return 1;
    }
}
