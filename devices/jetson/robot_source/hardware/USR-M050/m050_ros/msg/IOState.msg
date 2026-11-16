# Time this full IO read finished, including failed attempts.
builtin_interfaces/Time stamp
# Both arrays are meaningful only when valid is true. Failed reads contain zeros.
bool valid
# Array index 0 = channel 1, index 1 = channel 2.
bool[2] do_state
bool[2] di_state
# SDK result: 0 = success, negative = error (see m050/m050.h).
int32 error_code
string message
