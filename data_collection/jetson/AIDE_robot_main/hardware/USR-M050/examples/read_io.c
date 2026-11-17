#include <m050/m050.h>
#include <stdio.h>

int main(int argc, char **argv)
{
    m050_config config;
    m050_client *client = NULL;
    m050_error error;
    m050_io_state state;
    int rc;
    m050_config_default(&config);
    if (argc > 1) config.host = argv[1];
    rc = m050_create(&config, &client, &error);
    if (rc == M050_OK) rc = m050_read_io(client, &state, &error);
    if (rc == M050_OK) {
        printf("DO1=%u DO2=%u DI1=%u DI2=%u\n", (unsigned)state.do_state.ch1,
               (unsigned)state.do_state.ch2, (unsigned)state.di.ch1, (unsigned)state.di.ch2);
        /* Business actions, when needed (not executed by this example):
           rc = m050_set_do(client, 1, 1, NULL, &error); // DO1 close
           rc = m050_set_do(client, 1, 0, NULL, &error); // DO1 open
           rc = m050_set_do(client, 2, 1, NULL, &error); // DO2 close
           rc = m050_set_do(client, 2, 0, NULL, &error); // DO2 open
           Check rc and error.write_may_have_applied after EVERY call. */
    } else {
        fprintf(stderr, "M050 error %d: %s\n", rc, error.message);
    }
    m050_destroy(client);
    return rc == M050_OK ? 0 : 1;
}
